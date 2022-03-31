# Game Disruptor 2
The Game Disruptor 2.0 is a prototype designed to allow fully automated exploration of game variables across two balance objectives.
The tool was designed to be game-independent with a json-based inter-process communication (IPC) allowing for any game to be modified to interface with the tool. This approach was taken so that new games could be supported without needing to make changes to the Game Disruptor tool itself.

The pre-built version of the prototype comes with two games already set-up ready to balance, MicroRTS framework and the Disaster at Joadia Islands wargame.
The latest release can be found in the releases section of this repository here: https://github.com/TheJakeSnell/game-disruptor-2/releases

## Documentation
At the root of this repository exists two word documents and one video tutorial that provide an overview of the steps involved to setup and use the Game Disruptor tool.

### Game Disruptor 2 - Add Game Documentation.docx
This document outlines the necessary steps to add a new game to balance for the game disruptor tool and the requirements that a game must met in order to be added. The game disruptor 2.0 tool allows a game or wargame designer to balance their game without requiring any code changes to the tool itself. Instead, minor code changes will be needed to made to the game or simulator that will be used, this is to allow communication between the tool and the game. This limits the number of games a designer can use with the game disruptor tool, to the number of games that they have access to the source code or can request changes to be made on their behalf.

### Game Disruptor 2 - Balance Game Documentation.docx
This document outlines the simple steps to balance a game using the game disruptor tool. Before reading this document, please make sure you have setup your game correctly and followed the steps listed in the “Game Disruptor 2 - Add Game Documentation.docx” document.

### Game Disruptor 2 - Video Tutorial.mp4
This video tutorial visually covers the steps outlined in the "Game Disruptor 2 - Add Game Documentation.docx" with addition commentary as to why certain approaches were taken.
