import torch
from opensimplex import OpenSimplex
import random

class Wind:
    def __init__(self, dims: tuple, scale=0.1, seed=0, device="cpu"):
        """
        dims   : (nx, ny) for 2D or (nx, ny, nz) for 3D
        scale  : 노이즈 스케일 (작을수록 변동이 느려짐)
        seed   : 랜덤 오프셋용 시드
        """
        self.simp = OpenSimplex(seed=42)
        self.dims = dims
        self.ndim = len(dims)
        assert self.ndim in (2, 3), "dims must be 2D or 3D"

        self.scale = scale

        self.device = device

        # u,v,(w) 성분마다 다른 오프셋 적용
        g = torch.Generator().manual_seed(seed)
        self.offsets = torch.randint(0, 10_000, (3,), generator=g).tolist()

    def __call__(self, t: float) -> torch.Tensor:
        """
        시간 t에서 벡터장 생성
        2D: (2, ny, nx)
        3D: (3, nz, ny, nx)
        """
        if self.ndim == 2:
            nx, ny = self.dims
            field = torch.zeros((2, ny, nx), dtype=torch.float32, device = self.device)

            for j in range(ny):
                for i in range(nx):
                    x = i * self.scale
                    y = j * self.scale
                    z = 0.0

                    field[0, j, i] = self.simp.noise4(x + self.offsets[0],
                                             y + self.offsets[0],
                                             z + self.offsets[0],
                                             t * self.scale)
                    field[1, j, i] = self.simp.noise4(x + self.offsets[1],
                                             y + self.offsets[1],
                                             z + self.offsets[1],
                                             t * self.scale)
            return field

        elif self.ndim == 3:
            nx, ny, nz = self.dims
            field = torch.zeros((3, nz, ny, nx), dtype=torch.float32, device = self.device)

            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        x = i * self.scale
                        y = j * self.scale
                        z = k * self.scale

                        for c in range(3):
                            field[c, k, j, i] = self.simp.noise4(
                                x + self.offsets[c],
                                y + self.offsets[c],
                                z + self.offsets[c],
                                t * self.scale
                            )
            return field

class FastWind:
    def __init__(self, dims: tuple, scale=0.1, seed=0, device="cpu"):
        """
        dims   : (nx, ny) for 2D or (nx, ny, nz) for 3D
        scale  : 노이즈 스케일 (작을수록 변동이 느려짐)
        seed   : 랜덤 오프셋용 시드
        device : "cpu" or "cuda"
        """
        self.dims = dims
        self.ndim = len(dims)
        assert self.ndim in (2, 3), "dims must be 2D or 3D"
        self.scale = scale
        self.device = device

        # 각 성분(u,v,(w))마다 랜덤 오프셋
        g = torch.Generator(device=device).manual_seed(seed)
        self.offsets = torch.randint(0, 10_000, (3,), generator=g, device=device)

    def _noise4d(self, x, y, z, t):
        """
        vectorized 4D pseudo-simplex noise
        입력: x, y, z, t 모두 tensor
        """
        # pseudo-random hash 기반 noise
        def hash4(i, j, k, l):
            return torch.frac(torch.sin(i*12.9898 + j*78.233 + k*37.719 + l*24.823) * 43758.5453)

        # 좌표 정수/소수 분리
        xi, yi, zi, ti = torch.floor(x), torch.floor(y), torch.floor(z), torch.floor(t)
        xf, yf, zf, tf = x - xi, y - yi, z - zi, t - ti

        # 16개의 코너 hash
        n = 0
        for dx in [0,1]:
            for dy in [0,1]:
                for dz in [0,1]:
                    for dt in [0,1]:
                        h = hash4(xi+dx, yi+dy, zi+dz, ti+dt)
                        weight = (1 - dx + (2*dx-1)*xf) * (1 - dy + (2*dy-1)*yf) \
                               * (1 - dz + (2*dz-1)*zf) * (1 - dt + (2*dt-1)*tf)
                        n += h * weight
        return n * 2 - 1  # -1 ~ 1

    def __call__(self, t: float) -> torch.Tensor:
        """
        시간 t에서 벡터장 생성
        2D: (2, ny, nx)
        3D: (3, nz, ny, nx)
        """
        device = self.device
        if self.ndim == 2:
            nx, ny = self.dims
            x = torch.arange(nx, device=device) * self.scale
            y = torch.arange(ny, device=device) * self.scale
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            z = torch.zeros_like(grid_x)
            t_tensor = torch.full_like(grid_x, t * self.scale)

            field = torch.stack([
                self._noise4d(grid_x + self.offsets[0], grid_y + self.offsets[0], z + self.offsets[0], t_tensor),
                self._noise4d(grid_x + self.offsets[1], grid_y + self.offsets[1], z + self.offsets[1], t_tensor)
            ], dim=0)
            field = field.permute(0,2,1)  # (2, ny, nx)
            return field

        elif self.ndim == 3:
            nx, ny, nz = self.dims
            x = torch.arange(nx, device=device) * self.scale
            y = torch.arange(ny, device=device) * self.scale
            z = torch.arange(nz, device=device) * self.scale
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
            t_tensor = torch.full_like(grid_x, t * self.scale)

            field = torch.stack([
                self._noise4d(grid_x + self.offsets[0], grid_y + self.offsets[0], grid_z + self.offsets[0], t_tensor),
                self._noise4d(grid_x + self.offsets[1], grid_y + self.offsets[1], grid_z + self.offsets[1], t_tensor),
                self._noise4d(grid_x + self.offsets[2], grid_y + self.offsets[2], grid_z + self.offsets[2], t_tensor)
            ], dim=0)
            field = field.permute(0,2,1,3)  # (3, nz, ny, nx)
            return field

class NaturalWind:
    def __init__(self, dims: tuple, scale=0.1, seed=None, device="cpu", octaves=4):
        """
        dims   : (nx, ny) for 2D or (nx, ny, nz) for 3D
        scale  : 노이즈 스케일 (작을수록 느린 변화)
        seed   : 랜덤 시드
        device : "cpu" or "cuda"
        octaves: multi-octave noise 개수
        """
        self.dims = dims
        self.ndim = len(dims)
        assert self.ndim in (2, 3), "dims must be 2D or 3D"
        self.scale = scale
        self.device = device
        self.octaves = octaves

        # 성분별 오프셋
        if not seed:
            seed = random.randrange(0, 100)
        g = torch.Generator(device=device).manual_seed(seed)
        self.offsets = torch.randint(0, 10_000, (3,), generator=g, device=device)

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _noise4d(self, x, y, z, t):
        def hash4(i,j,k,l):
            # pseudo-random gradient
            return torch.frac(torch.sin(i*127.1 + j*311.7 + k*74.7 + l*59.2) * 43758.5453)
    
        xi, yi, zi, ti = torch.floor(x), torch.floor(y), torch.floor(z), torch.floor(t)
        xf, yf, zf, tf = x-xi, y-yi, z-zi, t-ti
    
        u, v, w, s = self._fade(xf), self._fade(yf), self._fade(zf), self._fade(tf)
    
        n = 0.0
        for dx in [0,1]:
            for dy in [0,1]:
                for dz in [0,1]:
                    for dt in [0,1]:
                        h = hash4(xi+dx, yi+dy, zi+dz, ti+dt)
                        wx = (1-u) if dx==0 else u
                        wy = (1-v) if dy==0 else v
                        wz = (1-w) if dz==0 else w
                        wt = (1-s) if dt==0 else s
                        n += h * wx * wy * wz * wt
        return n*2 - 1


    def _multi_octave(self, x, y, z, t):
        """
        여러 옥타브 합성
        """
        value = 0.0
        amp = 1.0
        freq = 1.0
        total_amp = 0.0
        for _ in range(self.octaves):
            value += amp * self._noise4d(x*freq, y*freq, z*freq, t*freq)
            total_amp += amp
            amp *= 0.5
            freq *= 2.0
        return value / total_amp

    def _div_free_2d(self, u, v):
        """2D divergence-free projection"""
        # FFT 기반 방법도 가능하지만 간단히 회전
        u_new = -v
        v_new = u
        return u_new, v_new

    def _div_free_3d(self, u, v, w):
        """3D divergence-free projection (approx)"""
        # 간단히 curl-like rotation
        u_new = v - w
        v_new = w - u
        w_new = u - v
        return u_new, v_new, w_new

    def __call__(self, t: float) -> torch.Tensor:
        device = self.device
        if self.ndim == 2:
            nx, ny = self.dims
            x = torch.arange(nx, device=device)*self.scale
            y = torch.arange(ny, device=device)*self.scale
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            z = torch.zeros_like(grid_x)
            t_tensor = torch.full_like(grid_x, t*self.scale)

            u = self._multi_octave(grid_x + self.offsets[0], grid_y + self.offsets[0], z + self.offsets[0], t_tensor)
            v = self._multi_octave(grid_x + self.offsets[1], grid_y + self.offsets[1], z + self.offsets[1], t_tensor)

            u, v = self._div_free_2d(u, v)
            field = torch.stack([u, v], dim=0).permute(0,2,1)  # (2, ny, nx)
            return field

        elif self.ndim == 3:
            nx, ny, nz = self.dims
            x = torch.arange(nx, device=device)*self.scale
            y = torch.arange(ny, device=device)*self.scale
            z = torch.arange(nz, device=device)*self.scale
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
            t_tensor = torch.full_like(grid_x, t*self.scale)

            u = self._multi_octave(grid_x + self.offsets[0], grid_y + self.offsets[0], grid_z + self.offsets[0], t_tensor)
            v = self._multi_octave(grid_x + self.offsets[1], grid_y + self.offsets[1], grid_z + self.offsets[1], t_tensor)
            w = self._multi_octave(grid_x + self.offsets[2], grid_y + self.offsets[2], grid_z + self.offsets[2], t_tensor)

            u, v, w = self._div_free_3d(u, v, w)
            field = torch.stack([u, v, w], dim=0).permute(0,2,1,3)  # (3, nz, ny, nx)
            return field

