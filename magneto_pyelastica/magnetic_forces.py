__doc__ = """Module implementation for external magnetic forces for magnetic Cosserat rods."""
__all__ = ["MagneticForces"]

from elastica.external_forces import NoForces
from elastica.rod.cosserat_rod import CosseratRod
from elastica._linalg import _batch_cross, _batch_matvec, _batch_norm
from magneto_pyelastica.magnetic_field import BaseMagneticField
import numpy as np
from typing import Union


class MagneticForces(NoForces):
    """
    磁力矩作用类：
    - external_magnetic_field.value(time) 返回 shape == (3,) 的均匀磁场；
    - magnetization_direction 在世界坐标系下给出，在初始化时转换到材料坐标系；
    - magnetization_density 可以是标量或 shape == (n_elems,) 的数组。
    """

    def __init__(
        self,
        external_magnetic_field: BaseMagneticField,
        magnetization_density: Union[float, int, np.ndarray],
        magnetization_direction: np.ndarray,
        rod_volume: np.ndarray,
        rod_director_collection: np.ndarray,
    ):
        super().__init__()

        # ==== 外磁场 ====
        self.external_magnetic_field = external_magnetic_field

        # ==== 体积和单元数 ====
        rod_volume = np.asarray(rod_volume, dtype=float)
        rod_n_elem = rod_volume.shape[0]

        # --------------------------------------------------
        # 1. 磁化方向（世界坐标系） -> (3, n_elems)
        # --------------------------------------------------
        mag_dir = np.asarray(magnetization_direction, dtype=float)

        if mag_dir.shape == (3,):
            # 全杆统一方向
            mag_dir = np.repeat(mag_dir[:, None], rod_n_elem, axis=1)  # (3, n_elems)
        elif mag_dir.shape == (rod_n_elem, 3):
            mag_dir = mag_dir.T  # (3, n_elems)
        elif mag_dir.shape == (3, rod_n_elem):
            pass  # 已经是 (3, n_elems)
        else:
            raise ValueError(
                f"Invalid magnetization direction shape {mag_dir.shape}, "
                f"expected (3,), (3, n_elems) or (n_elems, 3)."
            )

        # 归一化，防止零向量
        norm = _batch_norm(mag_dir)  # (n_elems,)
        if np.any(norm == 0):
            raise ValueError("Magnetization direction contains zero vector(s).")
        mag_dir /= norm  # (3, n_elems)

        # 世界坐标 -> 材料坐标
        mag_dir_material = _batch_matvec(rod_director_collection, mag_dir)  # (3, n_elems)

        # --------------------------------------------------
        # 2. 磁化强度 magnetization_density
        # --------------------------------------------------
        if np.isscalar(magnetization_density):
            mag_den = float(magnetization_density)  # 标量
        elif isinstance(magnetization_density, np.ndarray):
            mag_den = np.asarray(magnetization_density, dtype=float)
            if mag_den.shape != (rod_n_elem,):
                raise ValueError(
                    f"Invalid magnetization density shape {mag_den.shape}, "
                    f"expected scalar or (n_elems,)."
                )
        else:
            raise ValueError(
                "Invalid magnetization density: expected scalar or numpy array."
            )

        # 每个单元磁矩：M_i = (mag_den_i * volume_i) * dir_i
        scale = mag_den * rod_volume           # (n_elems,)
        self.magnetization_collection = mag_dir_material * scale  # (3, n_elems)

    # ================== 关键：函数签名 ==================
    def apply_torques(
        self,
        system: CosseratRod = None,
        time: np.float64 = 0.0,
        *args,
        **kwargs,
    ):
        """
        兼容 PyElastica 的调用方式：
        - 内部会调用 apply_torques(system=rod, time=time)
        这里直接用 system 这个参数作为杆对象。
        """
        rod = system
        if rod is None:
            raise ValueError("MagneticForces.apply_torques 需要提供 system 参数。")

        rod.external_torques += _batch_cross(
            self.magnetization_collection,
            # convert external_magnetic_field to local frame
            _batch_matvec(
                rod.director_collection,
                self.external_magnetic_field.value(time=time).reshape(
                    3, 1
                )  # broadcasting 3D vector
                * np.ones((rod.n_elems,)),
            ),
        )

    def apply_forces(
        self,
        system: CosseratRod = None,
        time: np.float64 = 0.0,
        *args,
        **kwargs,
    ):
        """
        同样接受 system 关键字参数，防止报 unexpected keyword argument。
        均匀磁场下只产生力矩，不产生合力，这里直接 return。
        """
        return
