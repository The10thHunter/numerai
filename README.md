#README 
This project, and its subsequent classes created from it, will use this file to explain features for later implementation. If this file does end up becoming public for any particular reason, great, someone else can read my thoughts in this very unstructured format. I'll try not to lose you. 

The model.py file contains the class torchMod which has built-in functions that simplify deployment of a model for me. It has a cudatest function, which will be very very useful for deploying on the PC quickly. This is super important cause you write all your code on an ARM based Mac, so flexibility is a good thing. And before anyone else asks... Yes, I know I'm pushing to master, yes I know its bad, no I don't plan on adding anyone else to this project and no I will not, unless incredibly desperate push from my PC. 

model.py's torchMod class includes 
-  @testing decorator - Outputs time-related data to the function executed. 
- forwardprop() - Takes in the initial set and trains the model. 
- backwards() - Semi-backprop function? Calls lossfunc() and then tells the nn to adjust biases and weights 
- lossfunc() - Loss function which will be packaged with the model.


