import numpy as np

def neuron_average_std(nn_activities, neuron, sim):
    # We get the neuron activities
    activity_e0 = sim.retina_mapper.flyvis_to_flygym(nn_activities[neuron][:][0])
    activity_e1 = sim.retina_mapper.flyvis_to_flygym(nn_activities[neuron][:][1])

    # We compute the standard deviation of the neuron per eye, for all ommatidia
    std_e0 = np.std(activity_e0)
    std_e1 = np.std(activity_e1)
    
    # We return the average stardard deviation for both eyes, for all ommatidia
    std = (std_e0 + std_e1) / 2.0 
    return std

def immobile_behavior():
    return np.array([0, 0])

def std_behavior(
    nn_activities,
    sim,
    adaptative = True,
    t4 = True,
    t5 = True,
    tm = False
):
    # Array that decides the direction that the fly will turn. 1.2 is the maximum value that gives
    # satisfying results
    turn_bias_base = np.array([-1.2, 1.2])

    # This value multiplies the output of the neurons to make sure that the fly turns fast enough.
    # It is empirically determined through an iterative process
    Kp = 1

    t4_std = 0
    t5_std = 0
    t4_coeff = 1
    t5_coeff = 1

    # Look at activities of the relevent neurons
    if (t4):
        t4a_std = neuron_average_std(nn_activities, "T4a", sim)
        t4b_std = neuron_average_std(nn_activities, "T4b", sim)
        t4_std = t4a_std - t4b_std
        Kp = 2

    if (t5):
        t5a_std = neuron_average_std(nn_activities, "T5a", sim)
        t5b_std = neuron_average_std(nn_activities, "T5b", sim)
        t5_std = t5a_std - t5b_std
        Kp = 2

    if (tm):    
        tm1_activity = neuron_average_std(nn_activities, "Tm1", sim)
        tm2_activity = neuron_average_std(nn_activities, "Tm2", sim)
        tm3_activity = neuron_average_std(nn_activities, "Tm3", sim)
        tm4_activity = neuron_average_std(nn_activities, "Tm4", sim)
        tm9_activity = neuron_average_std(nn_activities, "Tm9", sim)
        t4_coeff = tm3_activity
        t5_coeff = tm1_activity + tm2_activity + tm4_activity + tm9_activity
        Kp = 5
    
    t_std = Kp * (t4_coeff * t4_std + t5_coeff * t5_std)
    
    noise_threshold = 0.05

    if (np.abs(t_std) < noise_threshold):
        t_std = 0

    if (not(adaptative)):
        t_std = np.sign(t_std)
    
    if (t_std > 1):
        t_std = 1
    
    if (t_std < -1):
        t_std = -1

    turn_bias = turn_bias_base * t_std

    return turn_bias