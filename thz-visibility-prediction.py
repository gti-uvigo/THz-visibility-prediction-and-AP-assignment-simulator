# -*- coding: utf-8 -*-
"""
# NN prediction for AP visibility probability (i.e., not blocked and in coverage range)
#
"""

import sys
import glob
import time
import re
import numpy as np
import scipy as sp
import scipy.stats as st
import pandas as pd
import math
import random

import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


## Example values which must be consistent with the dataset
RANDOM_SEED = 0
NUM_BS = 121 # Number of BS in the simulation
NUM_USERS_ARRAY = [1, 2] # Number of users in the simulation
PROFILE_ARRAY = [4] # For having different sets of input parameters
N_STEPS_ARRAY = [1] # Number of previous time steps considered
MAX_TS_ARRAY = [ 100 ] # Maximum number of time slots considered in the dataset



# Auxiliar function for computing confidence intervals
def mean_confidence_interval_v2(data, confidence=0.95):
    if (min(data) == max(data)):
        m = min(data)
        h = 0
    else:
        a = 1.0*np.array(data)
        n = len(a)
        m, se = np.mean(a), st.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return '{:.3f} {:.3f} {:.3f}'.format(m, max(m-h, 0), m+h)


def trim_data(data, percent=5):  # percent=5 seems reasonable
    sorted_data = sorted(data)
    n = len(sorted_data)
    outliers = int(n*percent/100)
    trimmed_data = sorted_data[outliers: n-outliers]
    return trimmed_data


# Deprecated
def distance_seen(dist, randomness=False):
    if dist != -1 and randomness:
        dist = max(dist + 2*(random.random() - 0.5), 0.001)
    if dist > 10:
        dist = -1
    return dist

def get_pathloss(dist, randomness = False, pathloss_threshold = 55):
    # Pathloss model from https://ieeexplore.ieee.org/abstract/document/9135643
    # frequency = 0.14 THz
    d0 = 0.35 # m
    PLd0 = 25 # dB
    gamma = 2.117
    sigma = 0.5712

    if dist == -1:
        return -1

    pathloss = PLd0 + 10 * gamma * math.log10(dist / d0)
    
    if randomness:
        pathloss += float(np.random.normal(0, sigma, 1))

    if pathloss > pathloss_threshold:
        pathloss = -1

    return pathloss


def load_dataset(filename, max_ts, NUM_USERS, NUM_BS, CURRENT_USER_POSITION_ENABLED,
                 PREV_TS_CURRENT_USER_POSITION_ENABLED, OTHER_USERS_POSITION_ENABLED,
                 OTHER_USERS_ROTATION_ENABLED, PREV_TS_OTHER_USERS_POSITION_ENABLED,
                 PREV_TS_OTHER_USERS_ROTATION_ENABLED, CURRENT_USER_ROTATION_ENABLED,
                 PREV_TS_CURRENT_USER_ROTATION_ENABLED, NEARBY_ENABLED,
                 NEARBY_USERS_DISTANCE_THRESHOLD, PAST_AVAILABLE_TIME_ENABLED,
                 PREV_TS_DISTANCES_INCLUDED_EXPLICITLY_ENABLED, DIRECTION_ENABLED):
    # Read data from file
    print("\n    Parsing input file: {}".format(filename), file=sys.stderr)
    # input_data = {ts : {id : {"pos": pos, "distances": (distances)}}}
    input_data = {}
    allpos_x = {}
    allpos_y = {}
    allpos_z = {}
    allrot = {}
    prev_ts = None
    prev_ts_tmp = None
    with open(filename) as in_file:
        num_lines = 0
        for line in in_file:
            line = line.strip()
            splitted_line = line.split(";")
            ts = float(splitted_line[0])
            if prev_ts != None and ts <= prev_ts:
                ts = prev_ts + 0.2
            if ts > max_ts:
                break
            all_lines.append(line)
            id = int(splitted_line[1])
            pos = tuple(float(x) for x in (splitted_line[2][1:-1]).split(","))
            assert(len(pos) == 3)
            rotation = float(splitted_line[3])
            distances_seen = tuple(get_pathloss(float(x)) for x in splitted_line[4:])
            distances = tuple(x != -1 for x in distances_seen)
            assert(len(distances) == NUM_BS)
            past_available_time = [0 for _ in range(NUM_BS)]

            if prev_ts_tmp != ts:
                prev_ts = prev_ts_tmp

            for bs_id in range(NUM_BS):
                if prev_ts != None and distances[bs_id] == True:
                    past_available_time[bs_id] = input_data[prev_ts][id]["past_available_time"][bs_id] + (
                        ts - prev_ts)
                    #assert(ts - prev_ts > 0.15 and ts - prev_ts < 0.25)
                else:
                    past_available_time[bs_id] = 0

            num_lines += 1
            if ts not in input_data:
                input_data[ts] = {}
            if id not in input_data[ts]:
                input_data[ts][id] = {"pos": pos, "distances": distances, "rotation": rotation,
                                    "distances_seen": distances_seen, "past_available_time": past_available_time}

            if ts not in allpos_x:
                allpos_x[ts] = [-1 for _ in range(NUM_USERS)]
            if ts not in allpos_y:
                allpos_y[ts] = [-1 for _ in range(NUM_USERS)]
            if ts not in allpos_z:
                allpos_z[ts] = [-1 for _ in range(NUM_USERS)]
            if ts not in allrot:
                allrot[ts] = [-1 for _ in range(NUM_USERS)]
            allpos_x[ts][id] = pos[0]
            allpos_y[ts][id] = pos[1]
            allpos_z[ts][id] = pos[2]
            allrot[ts][id] = rotation
            prev_ts_tmp = ts
            #print("{} {}".format(prev_ts, ts))

    input_data_list = []
    prev_ts = None
    prev_prev_ts = None

    for ts in sorted(input_data.keys()):
        for id in sorted(input_data[ts].keys()):
            input_data_entry = []
            # User position
            if CURRENT_USER_POSITION_ENABLED:
                input_data_entry.append(input_data[ts][id]["pos"][0])
                input_data_entry.append(input_data[ts][id]["pos"][1])
                input_data_entry.append(input_data[ts][id]["pos"][2])
            if PREV_TS_CURRENT_USER_POSITION_ENABLED:
                if prev_ts != None:
                    input_data_entry.append(input_data[prev_ts][id]["pos"][0])
                    input_data_entry.append(input_data[prev_ts][id]["pos"][1])
                    input_data_entry.append(input_data[prev_ts][id]["pos"][2])

            # Other user positions
            if OTHER_USERS_POSITION_ENABLED or OTHER_USERS_ROTATION_ENABLED:
                other_users_position_rotation = []
                for id2 in input_data[ts]:
                    if id2 != id:
                        if OTHER_USERS_POSITION_ENABLED and not OTHER_USERS_ROTATION_ENABLED:
                            other_users_position_rotation.append(
                                (allpos_x[ts][id2], allpos_y[ts][id2], allpos_z[ts][id2]))
                        elif not OTHER_USERS_ROTATION_ENABLED and OTHER_USERS_ROTATION_ENABLED:
                            other_users_position_rotation.append(math.sin(allrot[ts][id2]))
                            other_users_position_rotation.append(math.cos(allrot[ts][id2]))
                        elif OTHER_USERS_POSITION_ENABLED and OTHER_USERS_ROTATION_ENABLED:
                            other_users_position_rotation.append(
                                (allpos_x[ts][id2], allpos_y[ts][id2], allpos_z[ts][id2],
                                 math.sin(allrot[ts][id2]), math.cos(allrot[ts][id2])))

                other_users_position_rotation.sort()

                for pos_rot in other_users_position_rotation:
                    for x in pos_rot:
                        input_data_entry.append(x)

            if PREV_TS_OTHER_USERS_POSITION_ENABLED or PREV_TS_OTHER_USERS_ROTATION_ENABLED:
                if prev_ts != None:
                    other_users_position_rotation = []
                    for id2 in input_data[prev_ts]:
                        if id2 != id:
                            if PREV_TS_OTHER_USERS_POSITION_ENABLED and not PREV_TS_OTHER_USERS_ROTATION_ENABLED:
                                other_users_position_rotation.append(
                                    (allpos_x[prev_ts][id2], allpos_y[prev_ts][id2], allpos_z[prev_ts][id2]))
                            elif not PREV_TS_OTHER_USERS_ROTATION_ENABLED and PREV_TS_OTHER_USERS_ROTATION_ENABLED:
                                other_users_position_rotation.append(
                                    (math.sin(allrot[prev_ts][id2])))
                                other_users_position_rotation.append(
                                    (math.cos(allrot[prev_ts][id2])))
                            elif PREV_TS_OTHER_USERS_POSITION_ENABLED and PREV_TS_OTHER_USERS_ROTATION_ENABLED:
                                other_users_position_rotation.append(
                                    (allpos_x[prev_ts][id2], allpos_y[prev_ts][id2], allpos_z[prev_ts][id2],
                                     math.sin(allrot[prev_ts][id2]), math.cos(allrot[prev_ts][id2])))

                    other_users_position_rotation.sort()

                    for pos_rot in other_users_position_rotation:
                        for x in pos_rot:
                            input_data_entry.append(x)

            # User rotation
            if CURRENT_USER_ROTATION_ENABLED:
                input_data_entry.append(math.sin(input_data[ts][id]["rotation"]))
                input_data_entry.append(math.cos(input_data[ts][id]["rotation"]))

            # User rotation
            if PREV_TS_CURRENT_USER_ROTATION_ENABLED:
                if prev_ts != None:
                    input_data_entry.append(math.sin(input_data[prev_ts][id]["rotation"]))
                    input_data_entry.append(math.cos(input_data[prev_ts][id]["rotation"]))

            # Nearby users
            if NEARBY_ENABLED:
                nearby_users = 0
                for id2 in input_data[ts]:
                    if id2 != id:
                        distance = math.sqrt(((input_data[ts][id]["pos"][0] - allpos_x[ts][id2]) ** 2) +
                                            ((input_data[ts][id]["pos"][1] - allpos_y[ts][id2]) ** 2) +
                                            ((input_data[ts][id]["pos"][2] - allpos_z[ts][id2]) ** 2))

                        if distance <= NEARBY_USERS_DISTANCE_THRESHOLD:
                            nearby_users += 1
                input_data_entry.append(nearby_users)

            if PAST_AVAILABLE_TIME_ENABLED:
                # NOTE: We cannot provide the time available from the current timestamp but the previous one
                if prev_ts != None:
                    # if id == 0:
                    #     print("prev_ts = {}, distances = {}, past_avail_time = {}".format(prev_ts, input_data[prev_ts][id]["distances_seen"], input_data[prev_ts][id]["past_available_time"]))
                    for time_available in input_data[prev_ts][id]["past_available_time"]:
                        input_data_entry.append(time_available)

            # DEBUG: Experimental for explicitly including the value of the prev ts
            if PREV_TS_DISTANCES_INCLUDED_EXPLICITLY_ENABLED:
                if prev_ts != None:
                    for distance in input_data[prev_ts][id]["distances_seen"]:
                        input_data_entry.append(distance)

            if DIRECTION_ENABLED:
                if prev_ts != None and prev_prev_ts != None:
                    for bs_id in range(NUM_BS):
                        distance_prev_prev = input_data[prev_prev_ts][id]["distances_seen"][bs_id]
                        distance_prev = input_data[prev_ts][id]["distances_seen"][bs_id]
                        # DEBUG EXPERIMENTAL: Try to introduce current direction rather than previous (e.g., assuming we use a gyroscope)
                        # distance_prev_prev = input_data[prev_ts][id]["distances_seen"][bs_id]
                        # distance_prev = input_data[ts][id]["distances_seen"][bs_id]
                        # if distance_prev != -1 and distance_prev_prev != -1:
                        input_data_entry.append(
                            distance_prev - distance_prev_prev)

            # AP visibility -> OUTPUT variable
            for distance in input_data[ts][id]["distances"]:
                input_data_entry.append(distance)
            input_data_list.append(input_data_entry)
        prev_prev_ts = prev_ts
        prev_ts = ts

    print("Parsed input file", file=sys.stderr)

    return input_data_list

# convert series to supervised learning (from: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
def series_to_supervised(data, n_in=1, n_out=1, diff_ts=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i * diff_ts))
        names += [('var%d(t-%d)' % (j+1, i * diff_ts)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i * diff_ts))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i * diff_ts))
                    for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


for MAX_TS in MAX_TS_ARRAY:
    for NUM_USERS in NUM_USERS_ARRAY:
        for n_steps in N_STEPS_ARRAY:
            for PROFILE in PROFILE_ARRAY:
                FILENAME = "thz_datasets/OK/SimData-{}-users.csv".format(NUM_USERS)
                OUTFILENAME = "thz_datasets/output-{}-users-{}-steps-{}-profile-test.csv".format(NUM_USERS, n_steps, PROFILE)

                print("\n    SCENARIO: max_ts = {}, users = {}, steps = {}, profile = {}".format(MAX_TS, NUM_USERS, n_steps,
                                                                                            PROFILE), file=sys.stderr)

                CURRENT_USER_POSITION_ENABLED = False
                OTHER_USERS_POSITION_ENABLED = False
                CURRENT_USER_ROTATION_ENABLED = False
                OTHER_USERS_ROTATION_ENABLED = False

                PREV_TS_CURRENT_USER_POSITION_ENABLED = False
                PREV_TS_OTHER_USERS_POSITION_ENABLED = False
                PREV_TS_CURRENT_USER_ROTATION_ENABLED = False
                PREV_TS_OTHER_USERS_ROTATION_ENABLED = False

                if (PROFILE == 1):
                    PREV_TS_CURRENT_USER_POSITION_ENABLED = True
                    PREV_TS_OTHER_USERS_POSITION_ENABLED = True
                    PREV_TS_CURRENT_USER_ROTATION_ENABLED = True
                    PREV_TS_OTHER_USERS_ROTATION_ENABLED = True
                elif (PROFILE == 2):
                    PREV_TS_CURRENT_USER_POSITION_ENABLED = True
                    PREV_TS_OTHER_USERS_POSITION_ENABLED = False
                    PREV_TS_CURRENT_USER_ROTATION_ENABLED = True
                    PREV_TS_OTHER_USERS_ROTATION_ENABLED = False
                elif (PROFILE == 3):
                    PREV_TS_CURRENT_USER_POSITION_ENABLED = True
                    PREV_TS_OTHER_USERS_POSITION_ENABLED = False
                    PREV_TS_CURRENT_USER_ROTATION_ENABLED = False
                    PREV_TS_OTHER_USERS_ROTATION_ENABLED = False
                elif (PROFILE == 4):
                    PREV_TS_CURRENT_USER_POSITION_ENABLED = False
                    PREV_TS_OTHER_USERS_POSITION_ENABLED = False
                    PREV_TS_CURRENT_USER_ROTATION_ENABLED = False
                    PREV_TS_OTHER_USERS_ROTATION_ENABLED = False

                NEARBY_ENABLED = False
                PREV_TS_DISTANCES_INCLUDED_EXPLICITLY_ENABLED = False # This set to false means distances/signal values
                                                                    # are not considered, but just visibility or not
                DIRECTION_ENABLED = False
                PAST_AVAILABLE_TIME_ENABLED = False


                # Parameters for the model
                # Distance threshold for nearby users (in meters)
                NEARBY_USERS_DISTANCE_THRESHOLD = 0.5

                n_features = NUM_BS  # AP visibility -> output variable
                n_features += (3 if CURRENT_USER_POSITION_ENABLED else 0)
                n_features += (((NUM_USERS - 1) * 3) if OTHER_USERS_POSITION_ENABLED else 0)
                n_features += (2 if CURRENT_USER_ROTATION_ENABLED else 0)
                n_features += (((NUM_USERS - 1) * 2) if OTHER_USERS_ROTATION_ENABLED else 0)
                n_features += (3 if PREV_TS_CURRENT_USER_POSITION_ENABLED else 0)
                n_features += (((NUM_USERS - 1) * 3) if PREV_TS_OTHER_USERS_POSITION_ENABLED else 0)
                n_features += (2 if PREV_TS_CURRENT_USER_ROTATION_ENABLED else 0)
                n_features += (((NUM_USERS - 1) * 2) if PREV_TS_OTHER_USERS_ROTATION_ENABLED else 0)
                n_features += (NUM_BS if PREV_TS_DISTANCES_INCLUDED_EXPLICITLY_ENABLED else 0)
                n_features += (1 if NEARBY_ENABLED else 0)
                n_features += (NUM_BS if DIRECTION_ENABLED else 0)
                n_features += (NUM_BS if PAST_AVAILABLE_TIME_ENABLED else 0)


                n_features_predict = NUM_BS  # AP visibility

                all_lines = []


                random.seed(RANDOM_SEED)

                input_data = load_dataset(FILENAME, MAX_TS, NUM_USERS, NUM_BS, CURRENT_USER_POSITION_ENABLED,
                    PREV_TS_CURRENT_USER_POSITION_ENABLED, OTHER_USERS_POSITION_ENABLED,
                    OTHER_USERS_ROTATION_ENABLED, PREV_TS_OTHER_USERS_POSITION_ENABLED,
                    PREV_TS_OTHER_USERS_ROTATION_ENABLED, CURRENT_USER_ROTATION_ENABLED,
                    PREV_TS_CURRENT_USER_ROTATION_ENABLED, NEARBY_ENABLED,
                    NEARBY_USERS_DISTANCE_THRESHOLD, PAST_AVAILABLE_TIME_ENABLED,
                    PREV_TS_DISTANCES_INCLUDED_EXPLICITLY_ENABLED, DIRECTION_ENABLED)

                input_data_df = pd.DataFrame(input_data)
                input_data = input_data_df.iloc[:, :].values
                input_data = input_data.astype('float32')

                reframed = series_to_supervised(input_data, n_steps, 1, NUM_USERS)


                input_data = reframed


                start_time = time.time()
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler_train = MinMaxScaler(feature_range=(0, 1))

                values = np.array(input_data)
                train_size = int(0.7 * len(input_data))
                validation_size = int(0.1 * len(input_data))

                train = values[:train_size, :]
                validation = values[train_size:(train_size + validation_size), :]
                test = values[(train_size + validation_size):, :]

                all_lines_test = all_lines[(-len(test)):]

                X = values[:, :-n_features_predict]
                X = scaler.fit_transform(X)
                y = values[:, -n_features_predict:]

                # split into input and outputs
                train_X, train_y = train[:, :-
                                        n_features_predict], train[:, -n_features_predict:]
                train_X = scaler_train.fit_transform(train_X)
                validation_X, validation_y = validation[:, :-
                                                        n_features_predict], validation[:, -n_features_predict:]
                validation_X = scaler_train.transform(validation_X)
                test_X, test_y = test[:, :-n_features_predict], test[:, -n_features_predict:]
                test_X = scaler_train.fit_transform(test_X)

                model = Sequential()

                model.add(Dense(units=1000, input_shape=(n_steps * n_features +
                        (n_features - n_features_predict),), activation='relu'))
                model.add(Dense(units=1000, activation="relu"))
                # Last layer needs to have 2 units since we are predicting two features
                model.add(Dense(n_features_predict, activation="sigmoid"))  # sigmoid # hard_sigmoid
                model.compile(loss='binary_crossentropy',
                            optimizer='sgd', metrics=['accuracy'])
                # fit network
                history = model.fit(train_X, train_y, epochs=750, batch_size=100,  # epochs=1000
                                    shuffle=False, validation_data=(validation_X, validation_y), verbose=False)  # validation_data=(test_X, test_y)


                ##########################################
                ### make a prediction for training set ###
                ##########################################
                X = train_X
                y = train_y

                yhat = model.predict(X)  # model.predict(X)


                # invert scaling for forecast
                inv_yhat = np.concatenate((X, yhat), axis=1)

                inv_yhat = inv_yhat[:, -n_features_predict:]

                y_reshaped = y.reshape(y.shape[0], y.shape[1])
                inv_y = np.concatenate((X, y_reshaped), axis=1)

                inv_y = inv_y[:, -n_features_predict:]

                # Convert float to bool
                inv_y = np.array([(x > 0.5) for x in inv_y])
                inv_yhat = np.array([(x > 0.5) for x in inv_yhat])

                inv_y = inv_y.reshape(inv_y.shape[0] * inv_y.shape[1], 1)
                inv_yhat = inv_yhat.reshape(inv_yhat.shape[0] * inv_yhat.shape[1], 1)

                # print(inv_y)
                # print(inv_yhat)

                precision = precision_score(inv_y, inv_yhat)
                print("Training precision: {:.3f}".format(precision), file=sys.stderr)
                recall = recall_score(inv_y, inv_yhat)
                print("Training recall: {:.3f}".format(recall), file=sys.stderr)
                f1 = f1_score(inv_y, inv_yhat)
                print("Training f1-score: {:.3f}".format(f1), file=sys.stderr)
                accuracy = accuracy_score(inv_y, inv_yhat)
                print("Training accuracy: {:.3f}".format(accuracy), file=sys.stderr)

                ######################################
                ### make a prediction for test set ###
                ######################################
                X = test_X
                y = test_y

                yhat = model.predict(X)  # model.predict(X)

                # invert scaling for forecast
                inv_yhat = np.concatenate((X, yhat), axis=1)

                inv_yhat = inv_yhat[:, -n_features_predict:]

                y_reshaped = y.reshape(y.shape[0], y.shape[1])
                inv_y = np.concatenate((X, y_reshaped), axis=1)

                inv_y = inv_y[:, -n_features_predict:]

                inv_yhat_probs = inv_yhat

                # Convert float to bool
                inv_y = np.array([(x > 0.5) for x in inv_y])
                inv_yhat = np.array([(x > 0.5) for x in inv_yhat])

                inv_y_test = inv_y
                inv_yhat_test = inv_yhat

                inv_y = inv_y.reshape(inv_y.shape[0] * inv_y.shape[1], 1)
                inv_yhat = inv_yhat.reshape(inv_yhat.shape[0] * inv_yhat.shape[1], 1)

                inv_y_test_reshaped = inv_y
                inv_yhat_test_reshaped = inv_yhat

                test_precision = precision_score(inv_y, inv_yhat)
                print("Test precision: {:.3f}".format(test_precision), file=sys.stderr)
                test_recall = recall_score(inv_y, inv_yhat)
                print("Test recall: {:.3f}".format(test_recall), file=sys.stderr)
                test_f1 = f1_score(inv_y, inv_yhat)
                print("Test f1-score: {:.3f}".format(test_f1), file=sys.stderr)
                test_accuracy = accuracy_score(inv_y, inv_yhat)
                print("Test accuracy: {:.3f}".format(test_accuracy), file=sys.stderr)


                inv_y = inv_y_test[NUM_USERS:]
                inv_y_no_reshaped = inv_y
                inv_y = inv_y.reshape(inv_y.shape[0] * inv_y.shape[1], 1)
                naive_y = inv_y_test[:-NUM_USERS]
                naive_y_no_reshaped = naive_y
                naive_y = naive_y.reshape(naive_y.shape[0] * naive_y.shape[1], 1)

                precision = precision_score(inv_y, naive_y)
                print("", file=sys.stderr)
                print("Naive test precision: {:.3f}".format(precision), file=sys.stderr)
                recall = recall_score(inv_y, naive_y)
                print("Naive test recall: {:.3f}".format(recall), file=sys.stderr)
                f1 = f1_score(inv_y, naive_y)
                print("Naive test f1-score: {:.3f}".format(f1), file=sys.stderr)
                accuracy = accuracy_score(inv_y, naive_y)
                print("Naive test accuracy: {:.3f}".format(accuracy), file=sys.stderr)

                # AVAILABILITY METRIC

                # NN-based: Pick the one with the highest probability
                chosen_bs_ids = [np.argmax(x) for x in inv_yhat_probs[NUM_USERS:]]
                #availability = [inv_y_test[NUM_USERS + i][chosen_bs_ids[i]] for i in range(len(chosen_bs_ids))]
                availability = [inv_y_no_reshaped[i][chosen_bs_ids[i]]
                                for i in range(len(chosen_bs_ids))]
                print("\nTest availability: {:.3f}".format(
                    sum(availability) / len(availability)), file=sys.stderr)
                test_availability = availability


                # Naive: Pick random one between the True ones in the previous interval
                chosen_bs_ids = [(random.choice([i for i in range(len(x)) if x[i] == True])
                                if True in x else 0) for x in naive_y_no_reshaped]
                availability = [inv_y_no_reshaped[i][chosen_bs_ids[i]]
                                for i in range(len(chosen_bs_ids))]
                print("Naive test availability: {:.3f}".format(
                    sum(availability) / len(availability)), file=sys.stderr)
                naive_availability = availability

                # Ideal: Pick a real True one
                chosen_bs_ids = [np.argmax(x) for x in inv_y_test]
                availability = [inv_y_test[i][chosen_bs_ids[i]]
                                for i in range(len(chosen_bs_ids))]
                print("Ideal test availability: {:.3f}".format(
                    sum(availability) / len(availability)), file=sys.stderr)
                ideal_availability = availability

                with open(OUTFILENAME, 'w') as outfile:
                    for i in range(len(all_lines_test)):
                        # To have binary output
                        # print("{};{}".format(all_lines_test[i], ";".join(
                        #     str(int(x)) for x in inv_yhat_test[i])), file=outfile)
                        # To have [0, 1] probability output
                        print("{};{}".format(all_lines_test[i], ";".join(
                            str(float(x)) for x in inv_yhat_probs[i])), file=outfile)

                # aggregate by timestamp
                test_availability_trimmed = test_availability[:(len(test_availability)-(len(test_availability) % NUM_USERS))]
                naive_availability_trimmed = naive_availability[:(len(naive_availability)-(len(naive_availability) % NUM_USERS))]
                ideal_availability_trimmed = ideal_availability[:(len(ideal_availability)-(len(ideal_availability) % NUM_USERS))]
                test_availability_aggregated = [sum(x)/len(x) for x in np.reshape(test_availability_trimmed, (-1, NUM_USERS))]
                naive_availability_aggregated = [sum(x)/len(x) for x in np.reshape(naive_availability_trimmed, (-1, NUM_USERS))]
                ideal_availability_aggregated = [sum(x)/len(x) for x in np.reshape(ideal_availability_trimmed, (-1, NUM_USERS))]
                test_availability_trimmed = test_availability[:(len(test_availability)-(len(test_availability) % 10))]

                inv_y_test_reshaped_trimmed = inv_y_test_reshaped[:(len(inv_y_test_reshaped)-(len(inv_y_test_reshaped) % (NUM_USERS * NUM_BS)))]
                inv_yhat_test_reshaped_trimmed = inv_yhat_test_reshaped[:(len(inv_yhat_test_reshaped)-(len(inv_yhat_test_reshaped) % (NUM_USERS * NUM_BS)))]
                inv_y_test_aggregated = np.reshape(inv_y_test_reshaped_trimmed, (-1, (NUM_USERS * NUM_BS)))
                inv_yhat_test_aggregated = np.reshape(inv_yhat_test_reshaped_trimmed, (-1, (NUM_USERS * NUM_BS)))

                test_precision_aggregated = [precision_score(y, yhat) for y, yhat in zip(inv_y_test_aggregated, inv_yhat_test_aggregated)]
                test_recall_aggregated = [recall_score(y, yhat) for y, yhat in zip(inv_y_test_aggregated, inv_yhat_test_aggregated)]
                test_f1_aggregated = [f1_score(y, yhat) for y, yhat in zip(inv_y_test_aggregated, inv_yhat_test_aggregated)]
                test_accuracy_aggregated = [accuracy_score(y, yhat) for y, yhat in zip(inv_y_test_aggregated, inv_yhat_test_aggregated)]

                test_precision_mci = mean_confidence_interval_v2(test_precision_aggregated)
                test_recall_mci = mean_confidence_interval_v2(test_recall_aggregated)
                test_f1_mci = mean_confidence_interval_v2(test_f1_aggregated)
                test_accuracy_mci = mean_confidence_interval_v2(test_accuracy_aggregated)
                test_availability_mci = mean_confidence_interval_v2(test_availability_aggregated)
                naive_availability_mci = mean_confidence_interval_v2(naive_availability_aggregated)
                ideal_availability_mci = mean_confidence_interval_v2(ideal_availability_aggregated)

                print("test_precision_mci = {}".format(test_precision_mci), file=sys.stderr)
                print("test_recall_mci = {}".format(test_recall_mci), file=sys.stderr)
                print("test_f1_mci = {}".format(test_f1_mci), file=sys.stderr)
                print("test_accuracy_mci = {}".format(test_accuracy_mci), file=sys.stderr)
                print("test_availability_mci = {}".format(test_availability_mci), file=sys.stderr)
                print("naive_availability_mci = {}".format(naive_availability_mci), file=sys.stderr)
                print("ideal_availability_mci = {}".format(ideal_availability_mci), file=sys.stderr)

                print("{} {} {} {} {} {} {} {} {} {} {} {}".format(PROFILE, NUM_USERS, NUM_BS, n_steps, MAX_TS,
                                                                   test_precision_mci, test_recall_mci, test_f1_mci,
                                                                   test_accuracy_mci, test_availability_mci,
                                                                   naive_availability_mci, ideal_availability_mci))
