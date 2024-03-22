### https://towardsdatascience.com/create-your-own-reinforcement-learning-environment-beb12f4151ef

# Modlue
import turtle

# Creating the window
win = turtle.Screen()       # Creating the screen
win.title('Paddle')         # Set the title
win.bgcolor('black')        # Set the background color
win.tracer(0)
win.setup(width = 600, height = 600)    # Set the width and height of the widow

# Create the Paddle
paddle = turtle.Turtle()    # Create a turtle object
paddle.shape('square')      # Select a square shape
paddle.speed(0)
paddle.shapesize(stretch_wid = 1, stretch_len = 5)      # Streach the length of the square by 5
paddle.penup()
paddle.color('white')       # Set the color to white
paddle.goto(0, -275)        # Palce the shape on the bottom of the screen

# Create the ball
ball = turtle.Turtle()      # Create a turtle object
ball.speed(0)
ball.shape('circle')        # Select a circle shape
ball.color('red')           # Set the color to white
ball.penup()
ball.goto(0,100)            # Place the shape in the middle

    
# Paddle Movement
def paddle_right():
    x = paddle.xcor()       # Get the x position of the paddle
    if x < 225:
        paddle.setx(x + 20) # Increment the position by 20 

def paddle_left():
    x = paddle.xcor()
    if x > -225:
        paddle.setx(x - 20)

# Keyboard control
win.listen()
win.onkey(paddle_right, 'Right')      # Call paddle_right on right arrow key
win.onkey(paddle_left, 'Left')        # Call paddle_left on left arrow key

# Initial velocities
ball.dx = 1               # Ball x-axis velocity
ball.dy = - 1             # Ball y-axis velocity

# Scorecard
hit, miss = 0, 0

score = turtle.Turtle()     # Create turtle object
score.speed(0)
score.color('white')        # Set the color to white
score.hideturtle()          # Hide the shape of the object
score.goto(0, 250)          # Set scorecard to upper middle of the screen
score.penup()
score.write('Hit: {} Missed: {}'.format(hit, miss), align = 'center', font = ('Courier', 24, 'normal'))

while True:

    
    # Ball-Walls collision
    if ball.xcor() > 290:   # If ball touches the right wall
        ball.setx(290)
        ball.dx *= -1       # Reverse the x-axis velocity
    
    if ball.xcor() < -290:  # If ball touches left wall
        ball.setx(-290)
        ball.dx *= -1       # Reverse the x-axis velocity
    
    if ball.ycor() > 290:   # If ball touches upper wall
        ball.sety(290)
        ball.dy *= -1       # Reverse the y-axis velocity
    
    # Ball-Paddle collision
    if abs(ball.ycor() + 250) < 2 and abs(paddle.xcor() - ball.xcor()) <55:
        ball.dy *= -1
        hit += 1
    
    # Ball-Ground collision
    if ball.ycor() < -290: # If the ball touches the ground
        ball.goto(0,100)
        ball.dx = 1
        miss += 1
        
    ball.setx(ball.xcor() + ball.dx)    # Update the ball's x-location using velocity
    ball.sety(ball.ycor() + ball.dy)    # Update the ball's y-location using velocity
    
    score.clear()
    score.write('Hit: {} Missed: {}'.format(hit, miss), align = 'center', font = ('Courier', 24, 'normal'))
    
    win.update()            # Show the screen continuously

def step(self, action):
    
    reward, done = 0, 0
    
    if action == 0:     # if action is 0, move paddlet left
        paddle_left()
        reward += -1    # reward of -1 for moving the paddle

    if action == 2:     # if action is 2, move paddlet right
        paddle_right()
        reward += -1    # reward of -1 for moving the paddle
    
    # run_frame()         # run the game for one frame
    
    # Creating the state vector
    state = [paddle.xcor(), ball.xcor(), ball.ycor(), ball.dx, ball.dt]
    
    return reward, state, done
    
#%%  
