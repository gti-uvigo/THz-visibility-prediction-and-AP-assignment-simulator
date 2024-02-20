#! /usr/bin/env/python3
"""
# AP assignment for THz networks
#
"""


import sys
import random
import math
import numpy as np
import scipy as sp
import scipy.stats as st


## Example values which must be consistent with the thz-visibility-prediction
RANDOM_SEED = 0
NUM_BS = 121
NUM_USERS_ARRAY = [1, 2]
PROFILE_ARRAY = [4]
N_STEPS_ARRAY = [1]

PREDICTION_ALGORITHMS = [ "ideal", "baseline", "baseline", "pred_availability" ]
ASSIGNMENT_ALGORITHMS = [ "baseline_previous", "global_baseline_hysteresis", "baseline_previous", "highest_probability_hysteresis_0.05" ]

# Experimental metric
REASSIGNMENT_LATENCY = 0.1  # 1 = 1 SLOT DURATION


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

def get_prob_availability(input_data, ts):
    # prob_availability = {id: {bs_id : [boolean]}}
    prob_availability = {}
    if ts != None:
        for id in input_data[ts]:
            prob_availability[id] = input_data[ts][id]["prob_availability"]
    return prob_availability


def get_past_available_time(input_data, ts):
    # past_available_time = {id: {bs_id : [boolean]}}
    past_available_time = {}
    if ts != None:
        for id in input_data[ts]:
            past_available_time[id] = input_data[ts][id]["past_available_time"]
    return past_available_time


def get_distances_seen(input_data, ts):
    # distances_seen = {id: {bs_id : [boolean]}}
    distances_seen = {}
    if (ts != None):
        for id in input_data[ts]:
            distances_seen[id] = [get_pathloss(input_data[ts][id]["distances"][bs_id]) for bs_id in range(
                len(input_data[ts][id]["distances"]))]

    return distances_seen


def predict_blockage(input_data, prev_ts, current_ts, algorithm):
    # DEBUG
    #print(predict_blockage_baseline(input_data, current_ts))
    #print(predict_blockage_file(input_data, current_ts))

    if algorithm == "ideal":
        return predict_blockage_baseline(input_data, current_ts)
    if algorithm == "baseline":
        return predict_blockage_baseline(input_data, prev_ts)
    if algorithm == "pred_availability":
        return predict_blockage_file(input_data, current_ts)
    else:
        print("Unknown prediction algorithm: {}".format(
            algorithm), file=sys.stderr)

    return None


def predict_blockage_baseline(input_data, prev_ts):
    # Same blockage as previous interval
    # prediction = {id: {bs_id : [boolean]}}
    prediction = {}
    if (prev_ts != None):
        for id in input_data[prev_ts]:
            prediction[id] = [False for _ in range(
                len(input_data[prev_ts][id]["distances"]))]
            for bs_id in range(len(input_data[prev_ts][id]["distances"])):
                bs_dist = get_pathloss(
                    input_data[prev_ts][id]["distances"][bs_id])
                if bs_dist == -1:
                    prediction[id][bs_id] = True

    return prediction


def predict_blockage_file(input_data, current_ts):
    # Read blockage prediction from file
    # prediction = {id: {bs_id : [boolean]}}
    prediction = {}
    if (current_ts != None):
        for id in input_data[current_ts]:
            # TODO: Try other thresholds lower than 0.5
            prediction[id] = [
                (x < 0.1) for x in input_data[current_ts][id]["prob_availability"]]

    return prediction


def assign_aps(prev_ap_assignment, blockage_prediction, algorithm, prev_distances, prev_available_time, prob_availability):
    if algorithm == "baseline_no_previous":
        return assign_aps_baseline(prev_ap_assignment, blockage_prediction, False)
    if algorithm == "baseline_previous":
        return assign_aps_baseline(prev_ap_assignment, blockage_prediction)
    if algorithm == "global_baseline_hysteresis":
        return assign_aps_global_baseline(prev_ap_assignment, prev_distances, distance_hysteresis_threshold=3)
    if algorithm == "baseline_previous_strongest":
        return assign_aps_baseline_previous_experimental(prev_ap_assignment, prev_distances, blockage_prediction, strongest = True)
    if algorithm == "baseline_previous_weakest":
        return assign_aps_baseline_previous_experimental(prev_ap_assignment, prev_distances, blockage_prediction, strongest = False)
    if algorithm == "longest_availability":
        return assign_aps_available_time(prev_ap_assignment, blockage_prediction, prev_available_time, True)
    if algorithm == "shortest_availability":
        return assign_aps_available_time(prev_ap_assignment, blockage_prediction, prev_available_time, False)
    if algorithm == "highest_probability":
        return assign_aps_highest_probability(prev_ap_assignment, blockage_prediction, prob_availability)
    if algorithm.startswith("highest_probability_hysteresis"):
        prob_hysteresis_threshold = None
        try:
            prob_hysteresis_threshold = float(algorithm.split("_")[-1])
        except:
            prob_hysteresis_threshold = 0.001

        return assign_aps_highest_probability_hysteresis(prev_ap_assignment, blockage_prediction, prob_availability, prob_hysteresis_threshold)
    else:
        print("Unknown assignment algorithm: {}".format(
            algorithm), file=sys.stderr)

    return None


def assign_aps_baseline(prev_ap_assignment, blockage_prediction, use_previous=True):
    # if previous AP for each user still available -> maintain
    # else -> assign random available AP

    # ap_assignment = {id: {bs_id}}
    ap_assignment = {}

    for id in blockage_prediction:
        if use_previous and (prev_ap_assignment != None and id in prev_ap_assignment and prev_ap_assignment[id] != None and
                            blockage_prediction != None and id in blockage_prediction and blockage_prediction[id][prev_ap_assignment[id]] == False):
            # If previous is available use this
            prev_bs_id = prev_ap_assignment[id]
            ap_assignment[id] = prev_bs_id
        else:
            # Random
            available_aps = []
            for bs_id in range(len(blockage_prediction[id])):
                if blockage_prediction[id][bs_id] == False:
                    available_aps.append(bs_id)
            assigned_ap = None
            if len(available_aps) > 0:
                assigned_ap = random.choice(available_aps)
            ap_assignment[id] = assigned_ap

    return ap_assignment


def assign_aps_highest_probability_hysteresis(prev_ap_assignment, blockage_prediction, prob_availability, prob_hysteresis_threshold=0.1):
    # print(past_available_times)
    # if previous AP for each user still available -> maintain
    # else -> assign AP with the highest available probability

    # ap_assignment = {id: {bs_id}}
    ap_assignment = {}

    for id in blockage_prediction:
        prev_bs_id = None
        if (prev_ap_assignment != None and id in prev_ap_assignment and prev_ap_assignment[id] != None and
            blockage_prediction != None and id in blockage_prediction and blockage_prediction[id][prev_ap_assignment[id]] == False):
            prev_bs_id = prev_ap_assignment[id]

        # Best time available (i.e., longest or shortest)
        assigned_ap = None
        best_prob_available = None
        for bs_id in range(len(blockage_prediction[id])):
            if blockage_prediction[id][bs_id] == False:
                # The AP is available
                #print("prob_availability({}) = {}".format(id, prob_availability[id]))
                prob_available = prob_availability[id][bs_id]
                if (assigned_ap == None or (prob_available > best_prob_available)):
                    assigned_ap = bs_id
                    best_prob_available = prob_available

        if prev_bs_id != None and (best_prob_available - prob_availability[id][prev_bs_id] <= prob_hysteresis_threshold):
            assigned_ap = prev_bs_id

        ap_assignment[id] = assigned_ap


    return ap_assignment


def assign_aps_highest_probability(prev_ap_assignment, blockage_prediction, prob_availability, use_previous=False):
    # print(past_available_times)
    # if previous AP for each user still available -> maintain
    # else -> assign AP with the highest available probability

    # ap_assignment = {id: {bs_id}}
    ap_assignment = {}

    for id in blockage_prediction:
        if use_previous and (prev_ap_assignment != None and id in prev_ap_assignment and prev_ap_assignment[id] != None and
                            blockage_prediction != None and id in blockage_prediction and blockage_prediction[id][prev_ap_assignment[id]] == False):
            # If previous is available use this
            prev_bs_id = prev_ap_assignment[id]
            ap_assignment[id] = prev_bs_id
        else:
            # Best time available (i.e., longest or shortest)
            best_ap = None
            best_prob_available = None
            for bs_id in range(len(blockage_prediction[id])):
                if blockage_prediction[id][bs_id] == False:
                    # The AP is available
                    #print("prob_availability({}) = {}".format(id, prob_availability[id]))
                    prob_available = prob_availability[id][bs_id]
                    if (best_ap == None or (prob_available > best_prob_available)):
                        best_ap = bs_id
                        best_prob_available = prob_available
            ap_assignment[id] = best_ap

    return ap_assignment


def assign_aps_available_time(prev_ap_assignment, blockage_prediction, past_available_times, longest=True):
    # print(past_available_times)
    # if previous AP for each user still available -> maintain
    # else -> assign random available AP

    # ap_assignment = {id: {bs_id}}
    ap_assignment = {}

    for id in blockage_prediction:
        if (prev_ap_assignment != None and id in prev_ap_assignment and prev_ap_assignment[id] != None and
                blockage_prediction != None and id in blockage_prediction and blockage_prediction[id][prev_ap_assignment[id]] == False):
            # If previous is available use this
            prev_bs_id = prev_ap_assignment[id]
            ap_assignment[id] = prev_bs_id
        else:
            # Best time available (i.e., longest or shortest)
            best_ap = None
            best_past_available_time = None
            for bs_id in range(len(blockage_prediction[id])):
                if blockage_prediction[id][bs_id] == False:
                    # The AP is available
                    past_available_time = past_available_times[id][bs_id]
                    if (best_ap == None or (longest and past_available_time > best_past_available_time) or
                            (not longest and past_available_time < best_past_available_time)):
                        best_ap = bs_id
                        best_past_available_time = past_available_time
            ap_assignment[id] = best_ap

    return ap_assignment


def assign_aps_baseline_previous_experimental(prev_ap_assignment, prev_distances, blockage_prediction, strongest = True):
    # if previous AP for each user still available -> maintain
    # else -> assign random available AP

    # ap_assignment = {id: {bs_id}}
    ap_assignment = {}

    for id in blockage_prediction:
        if (prev_ap_assignment != None and id in prev_ap_assignment and prev_ap_assignment[id] != None and
                            blockage_prediction != None and id in blockage_prediction and blockage_prediction[id][prev_ap_assignment[id]] == False):
            # If previous is available use this
            prev_bs_id = prev_ap_assignment[id]
            ap_assignment[id] = prev_bs_id
        else:
            # Random
            available_aps = []
            for bs_id in range(len(blockage_prediction[id])):
                if blockage_prediction[id][bs_id] == False:
                    available_aps.append(bs_id)
            assigned_ap = None
            if len(available_aps) > 0:
                # Strongest signal (i.e., shortest distance)
                best_distance = None
                assigned_ap = None
                for bs_id in available_aps:
                    if best_distance == None or best_distance == -1 or (prev_distances[id][bs_id] != -1 and ((strongest and prev_distances[id][bs_id] < best_distance) or (not strongest and prev_distances[id][bs_id] > best_distance))):
                        assigned_ap = bs_id
                        best_distance = prev_distances[id][bs_id]
            ap_assignment[id] = assigned_ap

    return ap_assignment

def assign_aps_global_baseline(prev_ap_assignment, prev_distances, distance_hysteresis_threshold = 3):
    # assign to closest AP with hysteresis
    #distance_hysteresis_threshold = 3  # in meters

    # ap_assignment = {id: {bs_id}}
    ap_assignment = {}

    for id in prev_distances:
        prev_bs_id = None
        if (prev_ap_assignment != None and id in prev_ap_assignment and prev_ap_assignment[id] != None and
                prev_distances != None and id in prev_distances and prev_distances[id][prev_ap_assignment[id]] != -1):
            prev_bs_id = prev_ap_assignment[id]

        # Strongest signal (i.e., shortest distance)
        shortest_distance = None
        assigned_ap = None
        for bs_id in range(len(prev_distances[id])):
            if shortest_distance == None or shortest_distance == -1 or (prev_distances[id][bs_id] != -1 and prev_distances[id][bs_id] < shortest_distance):
                assigned_ap = bs_id
                shortest_distance = prev_distances[id][bs_id]

        # DEBUG
        # if prev_bs_id != None and prev_bs_id != assigned_ap:
        #     print("prev_distances[id][prev_bs_id] = {}".format(prev_distances[id][prev_bs_id]))
        #     print("shortest_distance = {}".format(shortest_distance))
        if prev_bs_id != None and prev_distances[id][prev_bs_id] != -1 and (prev_distances[id][prev_bs_id] - shortest_distance <
                                    distance_hysteresis_threshold):
            assigned_ap = prev_bs_id

        ap_assignment[id] = assigned_ap

    return ap_assignment


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


def main():
    print("THz network blockage prediction simulator", file=sys.stderr)

    for NUM_USERS in NUM_USERS_ARRAY:
        for PROFILE in PROFILE_ARRAY:
            for N_STEPS in N_STEPS_ARRAY:
                # Set random seed
                random.seed(RANDOM_SEED)

                FILENAME = "thz_datasets/output-{}-users-{}-steps-{}-profile-test.csv".format(NUM_USERS, N_STEPS, PROFILE)

                # Read data from file
                print("Parsing input file: {}".format(FILENAME), file=sys.stderr)
                input_data = {}
                prev_ts = None
                prev_ts_tmp = None

                first_id_read = False

                with open(FILENAME) as in_file:
                    num_lines = 0
                    for line in in_file:
                        line = line.strip()
                        splitted_line = line.split(";")
                        ts = float(splitted_line[0])
                        if prev_ts != None and ts <= prev_ts:
                            ts = prev_ts + 0.2

                        id = int(splitted_line[1])
                        if (not first_id_read and id != 0):
                            continue
                        elif id == 0:
                            first_id_read = True
                        pos = tuple(float(x) for x in (splitted_line[2][1:-1]).split(","))
                        assert(len(pos) == 3)
                        rotation = float(splitted_line[3])
                        distances = tuple(float(x) for x in splitted_line[4:(4 + NUM_BS)])
                        prob_availability = None
                        if len(splitted_line) >= (4 + NUM_BS):
                            prob_availability = tuple(
                                float(x) for x in splitted_line[(4 + NUM_BS):(4 + 2 * NUM_BS)])
                        assert(len(distances) == NUM_BS)
                        past_available_time = [0 for _ in range(NUM_BS)]

                        if prev_ts_tmp != ts:
                            prev_ts = prev_ts_tmp

                        for bs_id in range(NUM_BS):
                            if prev_ts != None and distances[bs_id] != -1:
                                past_available_time[bs_id] = input_data[prev_ts][id]["past_available_time"][bs_id] + (
                                    ts - prev_ts)
                            else:
                                past_available_time[bs_id] = 0
                        num_lines += 1
                        if ts not in input_data:
                            input_data[ts] = {}
                        if id not in input_data[ts]:
                            input_data[ts][id] = {"pos": pos, "rotation": rotation, "distances": distances,
                                                "past_available_time": past_available_time, "prob_availability": prob_availability}
                        prev_ts_tmp = ts

                print("Parsed input file", file=sys.stderr)

                for prediction_algorithm, assignment_algorithm in zip(PREDICTION_ALGORITHMS, ASSIGNMENT_ALGORITHMS):

                    min_ts_diff = None
                    max_ts_diff = None
                    prev_ts = None
                    prev_ap_assignment = None
                    tp_array = []
                    fp_array = []
                    fn_array = []
                    tn_array = []
                    precision_array = []
                    recall_array = []
                    f1_array = []
                    availability_array = []
                    changes_array = []
                    latency_array = []

                    wait_slots = {}

                    for ts in input_data:
                        # Process one timestamp
                        # TODO: Delete
                        if prev_ts != None:
                            ts_diff = ts - prev_ts
                            if max_ts_diff == None or ts_diff > max_ts_diff:
                                max_ts_diff = ts_diff
                            if min_ts_diff == None or ts_diff < min_ts_diff:
                                min_ts_diff = ts_diff

                        # Predict input data
                        blockage_prediction = predict_blockage(
                            input_data, prev_ts, ts, prediction_algorithm)

                        # Evaluate prediction quality
                        for id in input_data[ts]:
                            predicted_values = None
                            if id not in blockage_prediction:
                                # No prediction for current user -> Random
                                #print("No blockage prediction in ts = {} for user = {}".format(ts, id))
                                predicted_values = [random.choice([False, True]) for _ in range(
                                    len(input_data[ts][id]["distances"]))]
                                blockage_prediction[id] = predicted_values
                            else:
                                #predicted_values = [random.choice([True, False]) for _ in range(len(input_data[ts][id]["distances"]))]
                                predicted_values = blockage_prediction[id]

                            real_values = [get_pathloss(
                                dist) == -1 for dist in input_data[ts][id]["distances"]]
                            tp = sum([(x == True and y == True)
                                    for x, y in zip(predicted_values, real_values)])
                            fp = sum([(x == True and y == False)
                                    for x, y in zip(predicted_values, real_values)])
                            fn = sum([(x == False and y == True)
                                    for x, y in zip(predicted_values, real_values)])
                            tn = sum([(x == False and y == False)
                                    for x, y in zip(predicted_values, real_values)])
                            precision = 1.0
                            recall = 1.0
                            f1 = 1.0
                            if tp != 0:
                                precision = (tp/(tp + fp))
                                recall = (tp/(tp+fn))
                                f1 = 2 * (precision * recall) / (precision + recall)
                            tp_array.append(tp)
                            fp_array.append(fp)
                            fn_array.append(fn)
                            tn_array.append(tn)
                            precision_array.append(precision)
                            recall_array.append(recall)
                            f1_array.append(f1)

                        prev_distances = get_distances_seen(input_data, prev_ts)
                        for id in input_data[ts]:
                            if id not in prev_distances:
                                prev_distances[id] = [-1 for _ in range(
                                    len(input_data[ts][id]["distances"]))]

                        prev_past_available_time = get_past_available_time(input_data, prev_ts)
                        for id in input_data[ts]:
                            if id not in prev_past_available_time:
                                prev_past_available_time[id] = [0 for _ in range(
                                    len(input_data[ts][id]["distances"]))]

                        prob_availability = get_prob_availability(input_data, ts)
                        for id in input_data[ts]:
                            if id not in prob_availability:
                                prob_availability[id] = [0 for _ in range(
                                    len(input_data[ts][id]["distances"]))]

                        # Assign
                        ap_assignment = assign_aps(
                            prev_ap_assignment, blockage_prediction, assignment_algorithm, prev_distances, prev_past_available_time, prob_availability)

                        # Evaluate assignment
                        num_available = 0
                        num_not_available = 0
                        num_changes = 0
                        for id in input_data[ts]:
                            if id not in wait_slots:
                                wait_slots[id] = 0
                            bs_id = ap_assignment[id]
                            available = not (bs_id == None or get_pathloss(input_data[ts][id]["distances"][bs_id]) == -1)
                            change = prev_ap_assignment != None and id in prev_ap_assignment and prev_ap_assignment[id] != None and prev_ap_assignment[id] != bs_id

                            lat_change = 0
                            if change:
                                num_changes += 1
                                lat_change = REASSIGNMENT_LATENCY

                            if not available:
                                # The chosen AP is not available
                                num_not_available += 1
                                wait_slots[id] += 1
                            else:
                                # The chosen AP is available
                                num_available += 1
                                for lat in range(wait_slots[id] + 1):
                                    latency_array.append(lat + lat_change)
                                wait_slots[id] = 0

                            #print("{}: -> {} {}".format(ts, bs_id, available))


                        availability_array.append(
                            num_available / (num_available + num_not_available))
                        changes_array.append(num_changes / len(input_data[ts]))

                        # Update values for the next interval
                        prev_ts = ts
                        prev_ap_assignment = ap_assignment

                    print("Prediction: {} - Assignment: {}".format(prediction_algorithm, assignment_algorithm), file=sys.stderr)

                    # print("DEBUG INFORMATION:")
                    # print("  Minimum timestamp difference: {}".format(min_ts_diff))
                    # print("  Maximum timestamp difference: {}".format(max_ts_diff))

                    # print("PREDICTION METRICS:")
                    # # print("  avg(tp_array) = {}".format(sum(tp_array)/len(tp_array)))
                    # # print("  avg(fp_array) = {}".format(sum(fp_array)/len(fp_array)))
                    # # print("  avg(fn_array) = {}".format(sum(fn_array)/len(fn_array)))
                    # # print("  avg(tn_array) = {}".format(sum(tn_array)/len(tn_array)))
                    # print("  avg(precision_array) = {}".format(
                    #     sum(precision_array)/len(precision_array)))
                    # print("  avg(recall_array) = {}".format(
                    #     sum(recall_array)/len(recall_array)))
                    # print("  avg(f1_array) = {}".format(sum(f1_array)/len(f1_array)))

                    # print("ASSIGNMENT METRICS:")
                    print("  avg(availability_array) = {}".format(
                        sum(availability_array)/len(availability_array)), file=sys.stderr)

                    print("  avg(changes_array) = {}".format(
                        sum(changes_array)/len(changes_array)), file=sys.stderr)
                    print("  avg(latency_array) = {}".format(
                        sum(latency_array)/len(latency_array)), file=sys.stderr)
                    print("", file=sys.stderr)

                    availability_mci = mean_confidence_interval_v2(availability_array)
                    changes_mci = mean_confidence_interval_v2(changes_array)
                    latency_mci = mean_confidence_interval_v2(latency_array)

                    prob_hysteresis_threshold = -1
                    if prediction_algorithm == "ideal" and assignment_algorithm == "baseline_previous":
                        ALGORITHM = "optimum"
                    if prediction_algorithm == "baseline" and assignment_algorithm == "baseline_previous":
                        ALGORITHM = "naive"
                    if prediction_algorithm == "baseline" and assignment_algorithm == "global_baseline_hysteresis":
                        ALGORITHM = "signal-hysteresis"
                    if prediction_algorithm == "pred_availability" and assignment_algorithm.startswith("highest_probability"):
                        ALGORITHM = "probability-hysteresis"
                        try:
                            prob_hysteresis_threshold = float(assignment_algorithm.split("_")[-1])
                        except:
                            prob_hysteresis_threshold = 0.001

                    print("{} {} {} {} {} {} {} {} {}".format(PROFILE, NUM_USERS, NUM_BS, N_STEPS, ALGORITHM, prob_hysteresis_threshold, availability_mci,
                                                        changes_mci, latency_mci))

if __name__ == "__main__":
    main()
