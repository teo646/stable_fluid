import numpy as np
import torch
from PIL import Image

from stable_fluid.fluid import Fluid

RESOLUTION = 150, 150, 400
DURATION = 10000

INFLOW_DURATION = 10000
INFLOW_VELOCITY = 1
INFLOW_COUNT = 1

chars = np.array(list(" .:-=+*#%@"))  # 밝기 단계에 대응

def main():
    print('Generating fluid solver, this may take some time.')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: ", device)
    fluid = Fluid(RESOLUTION, 'dye', viscosity = 0.01, device = device)

    inflow_velocity = torch.zeros_like(fluid.velocity, device=device)
    inflow_velocity[2, 70:80, 70:80, 0:8] += INFLOW_VELOCITY
    inflow_dye = torch.zeros(fluid.shape, device=device)
    inflow_dye[70:80, 70:80, 0:8] = 1.0

    dyes = []
    for f in range(DURATION):
        print(f'Computing frame {f + 1} of {DURATION}.')
        if f <= INFLOW_DURATION:
            fluid.velocity += inflow_velocity
            fluid.dye += inflow_dye
        fluid.step()
        dye_cpu = fluid.dye.cpu().numpy()
        dye_cpu[dye_cpu >= 1] = 1
        scale = dye_cpu[75]  * (len(chars) - 1)
        ascii_img = ["".join(chars[val] for val in row) for row in scale.astype(int)]

        print("\033[2J\033[H", end="")
        print("\n".join(ascii_img))
        #dyes.append(fluid.dye.cpu())

    #print('Saving simulation result.')
    #np.save("dyes.npy", np.array(dyes))

if __name__ == "__main__":
    main()
