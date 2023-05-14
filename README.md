# mario_nn

Decided to make this repository public, currently not working on it anymore but possibly a decent start for a mario neural network.
The main problem I found was the delay between input and output. 
The code interacts with the game in lua, which pushes the pressed control buttons to a .txt file, before a python file is called to call the neural network for what to do next.
This however causes a delay for 10-20 frames, which is a lot.
So this has to be fixed before the game can reasonably learn, ideally by programming the neural network in lua, but I had problems when I worked on that.
Anyway, still a decent start, and the neural network was able to reliably beat easy levels such as Yoshi's Island 2
