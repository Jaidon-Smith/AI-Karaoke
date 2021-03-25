# AI-Karaoke
This repository currently just consists of testing notebooks not usable code but I intend to reorganise this later.


The problem of aligning text with the audio where it was spoken is the same problem as forced alignment
I have found that Speech to Text systems trained with CTC loss can be also be used for forced alignment with little change.
My first experiments involved using Silero AI's Speech to text system to do this. It actually worked for singing which surprised me as I believe their model to be mostly trained on talking.

The next stage was essentially to recreate their work as Silero models do not include the Japanese language. I have been experiementing with transformer based models as well as convolution based.
