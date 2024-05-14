import numpy as np

contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

# fmt: off
cells = [
    "T1", "T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20",
    "Tm28", "Tm30", "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15",
    "TmY18"
]
# fmt: on

def get_eye_weighted_average(
    neural_activity
):
    print("not implemented yet")

def simple_std_behavior(
    nn_activities
):
    turn_bias_base = np.array([0, 0])
    # Look at activities of the relevent neurons
    t4a_activity_eye_one = sim.retina_mapper.flyvis_to_flygym(nn_activities["T4a"][:][0])
    t4a_activity_eye_one = sim.retina_mapper.flyvis_to_flygym(nn_activities["T4a"][:][1])
    t4b_activity_eye_two = sim.retina_mapper.flyvis_to_flygym(nn_activities["T4b"][:][0])
    t4b_activity_eye_two = sim.retina_mapper.flyvis_to_flygym(nn_activities["T4b"][:][1])

    # Compute the sandard deviations
    t4a_seddev_eye_one = np.std(t4a_activity_eye_one)
    t4a_seddev_eye_two = np.std(t4a_activity_eye_one)
    t4b_seddev_eye_one = np.std(t4b_activity_eye_two)
    t4b_seddev_eye_two = np.std(t4b_activity_eye_two)
    t4a_mean = (t4a_seddev_eye_one + t4a_seddev_eye_two) / 2.0
    t4b_mean = (t4b_seddev_eye_one + t4b_seddev_eye_two) / 2.0
    if(t4a_mean > t4b_mean):
        # Turn left
        turn_bias[0] = -1
        turn_bias[1] = 1
    else:
        # Turn right
        turn_bias[0] = 1
        turn_bias[1] = -1
    return turn_bias
    
def adaptaticve_std_behavior(
    nn_activities,
    sim
):
    turn_bias_base = np.array([-3, 3])
    # Look at activities of the relevent neurons
    t4a_activity_eye_one = sim.retina_mapper.flyvis_to_flygym(nn_activities["T4a"][:][0])
    t4a_activity_eye_one = sim.retina_mapper.flyvis_to_flygym(nn_activities["T4a"][:][1])
    t4b_activity_eye_two = sim.retina_mapper.flyvis_to_flygym(nn_activities["T4b"][:][0])
    t4b_activity_eye_two = sim.retina_mapper.flyvis_to_flygym(nn_activities["T4b"][:][1])

    # Compute the sandard deviations
    t4a_stddev_eye_one = np.std(t4a_activity_eye_one)
    t4a_stddev_eye_two = np.std(t4a_activity_eye_one)
    t4b_stddev_eye_one = np.std(t4b_activity_eye_two)
    t4b_stddev_eye_two = np.std(t4b_activity_eye_two)
    t4a_stddev_mean = (t4a_stddev_eye_one + t4a_stddev_eye_two) / 2.0
    t4b_stddev_mean = (t4b_stddev_eye_one + t4b_stddev_eye_two) / 2.0
    turning_coeff = t4a_stddev_mean - t4b_stddev_mean
    turn_bias = turn_bias_base * turning_coeff
    return turn_bias

def realistic_behavior(
    nn_activities,
    sim
):
    turn_bias_base = np.array([1, -1])
    
    # Relevant neurons with ommatidiae output from left eye
    # We take the absolute value because they can be polarized or depolarized
    tm1_activity_eye_one = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["Tm1"][:][0]))
    tm2_activity_eye_one = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["Tm2"][:][0]))
    tm3_activity_eye_one = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["Tm3"][:][0]))
    tm4_activity_eye_one = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["Tm4"][:][0]))
    tm9_activity_eye_one = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["Tm9"][:][0]))
    t5a_activity_eye_one = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["T5a"][:][0]))
    t5b_activity_eye_one = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["T5b"][:][0]))
    t4a_activity_eye_one = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["T4a"][:][0]))
    t4b_activity_eye_one = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["T4b"][:][0]))

    # Relevant neurons with ommatidiae output from right eye
    # We take the absolute value because they can be polarized or depolarized
    tm1_activity_eye_two = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["Tm1"][:][1]))
    tm2_activity_eye_two = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["Tm2"][:][1]))
    tm3_activity_eye_two = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["Tm3"][:][1]))
    tm4_activity_eye_two = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["Tm4"][:][1]))
    tm9_activity_eye_two = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["Tm9"][:][1]))
    t5a_activity_eye_two = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["T5a"][:][1]))
    t5b_activity_eye_two = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["T5b"][:][1]))
    t4a_activity_eye_two = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["T4a"][:][1]))
    t4b_activity_eye_two = np.absolute(sim.retina_mapper.flyvis_to_flygym(nn_activities["T4b"][:][1]))

    # We take the average of each eye
    tm1_activity = (tm1_activity_eye_one + tm1_activity_eye_two) / 2.0
    tm2_activity = (tm2_activity_eye_one + tm2_activity_eye_two) / 2.0
    tm3_activity = (tm3_activity_eye_one + tm3_activity_eye_two) / 2.0
    tm4_activity = (tm4_activity_eye_one + tm4_activity_eye_two) / 2.0
    tm9_activity = (tm9_activity_eye_one + tm9_activity_eye_two) / 2.0
    t5a_activity = (t5a_activity_eye_one + t5a_activity_eye_two) / 2.0
    t5b_activity = (t5b_activity_eye_one + t5b_activity_eye_two) / 2.0
    t4a_activity = (t4a_activity_eye_one + t4a_activity_eye_two) / 2.0
    t4b_activity = (t4b_activity_eye_one + t4b_activity_eye_two) / 2.0

    # Next we combine the outputs of different neurons to compute the optomotor response. The wey we do it is inspired
    # by connectomid data
    t5_coefficient = np.sum(tm1_activity + tm2_activity + tm4_activity + tm9_activity)
    t4_coefficient = np.sum(tm3_activity)
    t5_difference = np.sum(t5b_activity - t5a_activity)
    t4_difference = np.sum(t4b_activity - t4a_activity)
    optomotor_output = t5_coefficient * t5_difference + t4_coefficient * t4_difference
    #print(optomotor_output)
    return turn_bias_base * optomotor_output / 180000.0