"""
COT3502 Project 2
Daniel Romero
"""

# imports all the necessary libraries 
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy

# Creates a class named CreateVideo that takes care of plotting each frame and putting them into a video
class CreateVideo:
    def __init__(self, array, title, save, frames=200, interval=20, repeat=False, blit=True, show: bool = True):
        self.array = array
        self.title = title
        self.save = save
        self.frames = frames
        self.interval = interval
        self.repeat = repeat
        self.blit = blit
        self.show = show
        self.image = None
        self.setup()

    # Animate Function - runs for each frame of the video and plots the values the image
    def animate(self, i):
        self.image.set_array(self.array[:, :, i])  # Change to self.array[:, :, i]
        return [self.image]

    # Setup Function - establishes the settings of the plot, generates the animation, saves output as a video, and shows the plot to the user
    def setup(self):
        fig, ax = plt.subplots()
        self.image = ax.imshow(self.array[:, :, 0], interpolation="bicubic", cmap="jet")  # Change to self.array[:, :, i]
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(self.title)
        fig.tight_layout()

        output = animation.FuncAnimation(plt.gcf(), self.animate, frames=self.frames, interval=self.interval,
                                         repeat=self.repeat, blit=self.blit)

        print(">>> Generating Video")
        output.save(f"{self.save}.mp4", writer='ffmpeg', fps=30)
        print(">>> Video Generated")

        if self.show:
            plt.show()

# defines N, U, and V, where N = grid dimensions, and U and V represent concentration gradients for two chemicals
N = 512

U = numpy.ones((N,N))
V = numpy.zeros((N,N))

# find the mid point and the radius of disturbance
middle = N // 2
r = 10

# initializes the concentrations of the chemicals U and V by setting Initial Conditions
U[middle-r:middle+r, middle-r:middle+r] = 0.50
V[middle-r:middle+r, middle-r:middle+r] = 0.25

# defines the laplacian function to be used for the calculation of the diffusion
def laplacian(Z):
    return (numpy.roll(Z, 1, axis=0) + numpy.roll(Z, -1, axis=0) +
            numpy.roll(Z, 1, axis=1) + numpy.roll(Z, -1, axis=1) - 4 * Z)

# defines the update function that updates the values used for the simulation
def update(U, V, parameters):
    Du, Dv, F, k, dt = parameters
    u_diff = Du * laplacian(U)
    v_diff = Dv * laplacian(V)
    uvv = U * V**2
    U += dt * (u_diff - uvv + F * (1 - U))
    V += dt * (v_diff + uvv - (F + k) * V)
    return U, V

# establishes the parameters for the graphing of the Gray-Scott Model
# changing F and k results in different patterns in the simulation
parameters = (0.16, 0.08, 0.022, 0.05, 1.0)  # Example: Du, Dv, F, k, dt
steps = 1000 # amount of steps in the simulation / 5,000 steps makes a 33 second video / 200,000 steps makes a 22:13 minute video
frame_interval = 5 # divides the steps so that every 5th step gets used as a frame in the video

# creates and stores frames for the video
U_frame = numpy.zeros((N, N, int(steps/frame_interval)))
for i in range(steps):
    U, V = update(U, V, parameters)
    if i % frame_interval == 0:
        U_frame[:, :, i // frame_interval] = U
        
# creates and displays video from the data of the simulation
video = CreateVideo(U_frame, "Gray-Scott Model", f"f{str(parameters[2])[2:]}k{str(parameters[3])[2:]}", frames=steps, interval=frame_interval)
