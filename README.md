# Mini project: Explore how a connectome-constrained visual system network (Lappalainen et al.,bioRxiv 2023) can be used for a visuomotor task like optomotor turning from Week 4
This repository contains the code submitted for the "BIOEN-456: Controlling behavior in animals and robots" course. It is written by the group 4, comprised of Viva Berlenghi (324410), Florian David (375252) and Nils Antonovitch (310410).

## Contents of the repository
Here are the contents of the different files and folders:

- connectome_arenas.py: contains the code of the implementation of the two arenas that are used. One for the optomotor visual stimulus, the other for the looming visual stimulus;
- connectome_behavior.py: contains the code of the controller that dictates the movements of the fly in response to the optomotor visual stimulus.
- Connectome_with_arena.ipynb: contains the code necessary to generate all the simulations and .pkl files used for our miniproject;
- pkl_visualization.ipynb: contains the code needed to generate the plots seen in our report;
- outputs: contains the videos of all our simulations.

It is important to note that the __only files that should be run directly are the jupyter notebooks.__

## Submited vides and naming conventions
In our submission, we modified the names of the generated vides to make them easier to understand. In the "outputs" folder here, their names follow a different naming convention. Here is the structure of the names:

[A]_terrain_[A]_speed_[A].mp3

where:

- [A] is replaced by the controller used;
- [B] is replaced by the arena used;
- [C] is replaced by the speed given as a parameter to the used arena.

## How to generate the plots seen in our report
The code in the "pkl_visualization.ipynb" file is used to generate the plots that are in our report. To do so, it is important to have the data generated by the simulations in .pkl format and place it in the "outputs" folder. The names of the axis and the labels of the plots have to be changed manually.

## Where to find data generated by the code
All the .pkl files are too heavy to be submitted directly. They can be found at this link: https://drive.google.com/drive/folders/1IFcDRl41e9_0pbnqwAsV38HoEvjZ2gtM?usp=drive_link. They follow a similar naming conventions to the one described in the previous section. __All the plots that we generated for our report are generated using this data.__
