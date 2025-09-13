import torch
import torch.nn.functional as F
import torch.fft as fft
import math

class Fluid:
    def __init__(self, shape, *quantities, device='cuda'):
        #should be z(optional), y, x order.
        self.shape = shape
        self.dimensions = len(shape)
        self.device = device

        # velocity field (dimensions, Nz, Ny, Nx)
        # dimensions is x, y, z order.
        self.velocity = torch.zeros((self.dimensions, *shape), dtype=torch.float32, device=device)
        
        self.quantities = quantities
        for q in quantities:
            setattr(self, q, torch.zeros(shape, dtype=torch.float32, device=device))

        # grid indices
        indices = [torch.arange(s, device=device) for s in shape]
        mesh = torch.meshgrid(*indices, indexing='ij')
        self.indices = torch.stack(mesh[::-1])

    def advect(self, field, dt=1.0, filter_epsilon=1e-2, mode='border'):
        """
        Backward advection using trilinear interpolation on GPU, with minimal memory overhead.
        """
        device = field.device

        # Compute backward advection map
        # (3, z, y, x) shape
        advection_map = self.indices - self.velocity * dt 

        # Normalize to [-1,1] for grid_sample
        advection_map_norm = torch.empty_like(advection_map)
        
        for i in range(self.dimensions):
            size = field.shape[-(i+1)]
            advection_map_norm[i] = 2.0 * advection_map[i] / (size - 1) - 1.0

        #(1, Nz, Ny, Nx, 3)
        grid = advection_map_norm.permute(list(range(1, self.dimensions + 1)) + [0]).unsqueeze(0)

        field_unsq = field.unsqueeze(0).unsqueeze(0)  # (1, 1, Nz, Ny, Nx)

        # Trilinear interpolation
        advected = F.grid_sample(
            field_unsq, grid, align_corners=True,
            mode='bilinear', padding_mode=mode
        )

        # Blend with original field
        result = advected.squeeze(0).squeeze(0) * (1 - filter_epsilon) + field * filter_epsilon

        return result
        
    @staticmethod
    def poisson_fft(rhs, h=1.0):
        """
        Solve Laplacian p = rhs with periodic BC using FFT.
        Works for arbitrary dimensions (2D, 3D, ...).
        rhs: (...,), float32 cuda
        """
        dims = tuple(range(rhs.ndim))   # 모든 차원에서 FFT
        rhs_hat = fft.rfftn(rhs, dim=dims)

        # 각 축마다 wave numbers 생성
        ks = []
        for i, size in enumerate(rhs.shape):
            if i == rhs.ndim - 1:  # 마지막 축만 rfftfreq
                k = torch.fft.rfftfreq(size, d=h, device=rhs.device) * 2 * math.pi
            else:
                k = torch.fft.fftfreq(size, d=h, device=rhs.device) * 2 * math.pi
            ks.append(k)

        # meshgrid 만들기
        grids = torch.meshgrid(*ks, indexing="ij")

        # 일반화된 eigenvalue (라플라시안의 symbol)
        lam = torch.zeros_like(grids[0])
        for g in grids:
            lam += g**2

        lam[tuple([0] * rhs.ndim)] = 1.0  # DC 보호
        p_hat = -rhs_hat / lam
        p_hat[tuple([0] * rhs.ndim)] = 0.0

        # 역 FFT
        p = fft.irfftn(p_hat, s=rhs.shape, dim=dims)
        return p

    def get_divergence(self):
        """
        Compute divergence of vector field.
        velocity shape:
            2D: (2, Ny, Nx)
            3D: (3, Nz, Ny, Nx)
        """
        sum_ = torch.zeros_like(self.velocity[0])
        for i in range(self.dimensions):
            grad = torch.gradient(self.velocity[i], dim= -(i + 1))[0]
            sum_ += grad
        return sum_

    def gradient(self, field):
        list_ = []
        for i in range(self.dimensions):
            list_.append(torch.gradient(field, dim=-(i + 1))[0])
        return torch.stack(list_)

    def step(self, dt=1.0):
        # 1. Advect velocity
        for d in range(self.dimensions):
            self.velocity[d] = self.advect(self.velocity[d], self.velocity, dt)
        
        # 3. Pressure projection
        div = self.get_divergence()
        pressure = self.poisson_fft(div)
        grad_p = self.gradient(pressure)
        self.velocity -= grad_p
        if(torch.mean(torch.abs(self.get_divergence())) > 0.001):
            print("something went wrong. divergence too high.")
            print(torch.mean(torch.abs(self.get_divergence())))

        for q in self.quantities:
            setattr(self, q, self.advect(getattr(self, q), dt))
            
        return div, pressure
        
        
        