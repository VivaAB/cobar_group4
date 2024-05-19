import numpy as np

def immobile_behavior():
    return np.array([0, 0])

def simple_std_t4_behavior(
    nn_activities,
    sim
):
    # Array that decides the direction that the fly will turn
    turn_bias = np.array([0, 0])

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

    # Here the turning bias is not adaptative : if a threshold is crossed, the fly turns at a constant rate
    if(t4a_mean > t4b_mean):
        # Turn left
        turn_bias[0] = -1
        turn_bias[1] = 1
    else:
        # Turn right
        turn_bias[0] = 1
        turn_bias[1] = -1
    return turn_bias
    
def adaptative_std_t4_behavior(
    nn_activities,
    sim
):
    # Array that decides the direction that the fly will turn. It has values equal to 3 because the turning coefficient
    # is always small
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

    # Compute a turinig coefficient from the standard deviations and deduce the turning bias
    turning_coeff = t4a_stddev_mean - t4b_stddev_mean
    turn_bias = turn_bias_base * turning_coeff
    return turn_bias

def realistic_proportional_behavior(
    nn_activities,
    sim
):
    # Array that decides the direction that the fly will turn. The fly will turn right if the 
    # optomotor_output is greater than one
    turn_bias_base = np.array([1, -1])

    # This value, similar to the Kp in a P controller, is empirically determined through an iterative process:
    # it normalizes the output from the neurons and makes sure it can be exploited to compute the turning bias
    Kp = 180000.0
    
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

    # Next we combine the outputs of different neurons to compute the optomotor response. The way we do it is inspired
    # by connectomid data
    t5_coefficient = np.sum(tm1_activity + tm2_activity + tm4_activity + tm9_activity)
    t4_coefficient = np.sum(tm3_activity)
    t5_difference = np.sum(t5b_activity - t5a_activity)
    t4_difference = np.sum(t4b_activity - t4a_activity)
    optomotor_output = t5_coefficient * t5_difference + t4_coefficient * t4_difference
    return turn_bias_base * optomotor_output / Kp

def neuron_average_std(nn_activities, neuron, sim):
    
    # We get the neuron activities
    activity_e0 = sim.retina_mapper.flyvis_to_flygym(nn_activities[neuron][:][0])
    activity_e1 = sim.retina_mapper.flyvis_to_flygym(nn_activities[neuron][:][1])

    # We compute the standard deviation of the neuron per eye, for all ommatidia
    std_e0 = np.std(activity_e0)
    std_e1 = np.std(activity_e1)
    std = (std_e0 + std_e1) / 2.0
    
    # We return the average stardard deviation for both eyes, for all ommatidia
    return std

def simple_std_t45_behavior(nn_activities, sim):
    turn_bias_base = np.array([-1.5, 1.5])
    t4a_std = neuron_average_std(nn_activities, "T4a", sim)
    t4b_std = neuron_average_std(nn_activities, "T4b", sim)
    t5a_std = neuron_average_std(nn_activities, "T5a", sim)
    t5b_std = neuron_average_std(nn_activities, "T5b", sim)
    
    t4_std = t4a_std - t4b_std
    t5_std = t5a_std - t5b_std

    t_std = t4_std + t5_std

    noise_threshold = 0.1

    if np.abs(t_std) < noise_threshold:
        t_std = 0

    t_std = np.sign(t_std)
    
    turn_bias = turn_bias_base * t_std
    
    return turn_bias

def adaptative_std_t45_behavior(nn_activities, sim):
    # Array that decides the direction that the fly will turn. It has values equal to 4 because the turning coefficient
    # is always small
    turn_bias_base = np.array([-4, 4])

    t4a_std = neuron_average_std(nn_activities, "T4a", sim)
    t4b_std = neuron_average_std(nn_activities, "T4b", sim)
    t5a_std = neuron_average_std(nn_activities, "T5a", sim)
    t5b_std = neuron_average_std(nn_activities, "T5b", sim)
    
    t4_std = t4a_std - t4b_std
    t5_std = t5a_std - t5b_std

    t_std = t4_std + t5_std

    noise_threshold = 0.1

    if np.abs(t_std) < noise_threshold:
        t_std = 0
    
    turn_bias = turn_bias_base * t_std
    return turn_bias