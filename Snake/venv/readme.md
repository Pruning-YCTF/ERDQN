# AI Snake

This project created using python. Snake game was created using pygame library, NN using pyTorch.  
You can easily test it just by completing following steps:  


## Test project:
            * git clone https://github.com/kinfi4/AISnake.git
            * cd Snake
            * python3 -m venv env && source env/bin/activate
            * pip install -r requirements.txt
            * python test.py model/model.trained.pth

--------------
#### This project has two main scripts: train.py and test.py

### train.py: 
    Trains the model, visualize the snake and plots its scores and mean of its scores.  
    This script takes two optional arguments: 
        -f (--filename) <path to file> - file where script will score trained NN,
            default: ./model/model.pth
        -s (--short_form) - if is it set, scripts will not show you plots

    Examples:
        python train.py ./model/model.trained.pth -- train model and save it the specific path 
        python train.py ./model/model.trained.pth -s -- train model without plotting
        
        
### test.py
    Tests the model, just visualize the snake playing.
    This script takes one required argument and one optional:
        model_file - required positional arguments - path to the trained model
        -s (--speed) - optional argument, using it you can specify speed of the snake

    Examples:
        python test.py ./model/model.trained.pth
        python test.py ./model/model.trained.pth -s 100 -- play game with speed 100

------------------------------------------

## Architecture of NN
NN has three layers: input, hidden and output with (11, 256, 3) neurons respectively.  
Input layer gets game state, and build prediction based on it.  
Input later consists of 11 numbers:   
* Does the snake has danger straight, right or left from her (3 numbers)
* Direction of snake movement, right, down, left or up (4 numbers)
* Place of food, left, right, down and up from snake (4 numbers)

------------------------------------

## Screenshots:
<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
    <img src="https://github.com/kinfi4/AISnake/blob/main/docs/screenshots/graph.png?raw=true" width="">
    <img src="https://github.com/kinfi4/AISnake/blob/main/docs/screenshots/snake1.png?raw=true">
</div>

-------------------------------------------------------------------------------------

@kinfi4