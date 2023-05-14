# mario_nn

Decided to make this repository public, currently not working on it anymore but possibly a decent start for a mario neural network.
The main problem I found was the delay between input and output. 
The code interacts with the game in lua, which pushes the pressed control buttons to a .txt file and a screenshot every 10 frames, before a python file is called to call the neural network for what to do next.
This however causes a delay for 10-20 frames, which is a lot.
So this has to be fixed before the game can reasonably learn, ideally by programming the neural network in lua, but I had problems when I worked on that.
Anyway, still a decent start, and the neural network was able to reliably beat easy levels such as Yoshi's Island 2

# SETUP

Quick and dirty guide how to set it up, some more information can probably be seen in SethBlings guide for his MariFlow setup: ([MariFlow](https://docs.google.com/document/d/1p4ZOtziLmhf0jPbZTTaFxSKdYqE91dYcTNqTVdd6es4/edit#heading=h.syblbftlk25z))

1. Download BizHawk and put this repository in the Lua folder as a folder named 'mario_nn' 
2. Download the required python libraries and make sure you can execute `python watch.py`
3. Open BizHawk and open a Super Mario World ROM
4. Make savestate and save it as savestates/CustomState1.state at the start of the level you want to play
5. Open Tools > Lua Console
6. Optional: Set some variables in the console popup
7. Click Start, mario should now jump around, restart the level upon death/idle, and improve over time!

![screenshot of the popup](Media/popup%20screenshot.png?raw=True)
Experiment ID: give ID of experiment to seperate different runs
Continue last model: Set to true if you want to continue
Use weighted data: Use a dataloader weighted to get a balanced amount of x_speed values
Objectives: which objectives to optimize the model for
Use previous n: How many frames to look for in the past
