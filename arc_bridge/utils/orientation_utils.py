import numpy as np

DTYPE = np.float32


class Quaternion:
    def __init__(self, w:float=1, x:float=0, y:float=0, z:float=0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self._norm = np.sqrt(self.w*self.w+self.x*self.x+self.y*self.y+self.z*self.z, dtype=DTYPE)

    def to_numpy(self):
        """convert to an (4,1) numpy array"""
        return np.array([self.w,self.x,self.y,self.z], dtype=DTYPE)

    def unit(self):
        """return the unit quaternion"""
        return Quaternion(self.w/self._norm,self.x/self._norm,self.y/self._norm,self.z/self._norm)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def reverse(self):
        """return the reverse rotation representation as the same as the transpose op of rotation matrix"""
        return Quaternion(-self.w,self.x,self.y,self.z)

    def inverse(self):
        return Quaternion(self.w/(self._norm*self._norm),-self.x/(self._norm*self._norm),-self.y/(self._norm*self._norm),-self.z/(self._norm*self._norm))

    def __repr__(self):
        return '['+str(self.w)+', '+str(self.x)+', '+str(self.y)+', '+str(self.z)+']'


def quat_to_rpy(q:Quaternion) -> np.ndarray:
    """
    Convert a quaternion to RPY. Return
    angles in (roll, pitch, yaw).
    """
    rpy = np.zeros(3, dtype=DTYPE)
    as_ = np.min([-2.*(q.x*q.z-q.w*q.y),.99999])
    # roll
    rpy[0] = np.arctan2(2.*(q.y*q.z+q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
    # pitch
    rpy[1] = np.arcsin(as_)
    # yaw
    rpy[2] = np.arctan2(2.*(q.x*q.y+q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
    return rpy


def quat_to_rot(q:Quaternion) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix. This matrix represents a
    coordinate transformation into the frame which has the orientation specified
    by the quaternion
    """
    e0 = q.w
    e1 = q.x
    e2 = q.y
    e3 = q.z
    R = np.array([1 - 2 * (e2 * e2 + e3 * e3), 2 * (e1 * e2 - e0 * e3),
                  2 * (e1 * e3 + e0 * e2), 2 * (e1 * e2 + e0 * e3),
                  1 - 2 * (e1 * e1 + e3 * e3), 2 * (e2 * e3 - e0 * e1),
                  2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1),
                  1 - 2 * (e1 * e1 + e2 * e2)],
                  dtype=DTYPE).reshape((3,3))
    return R


def rpy_to_quat(rpy) -> Quaternion:
    cy = np.cos(rpy[2] * 0.5)
    sy = np.sin(rpy[2] * 0.5)
    cp = np.cos(rpy[1] * 0.5)
    sp = np.sin(rpy[1] * 0.5)
    cr = np.cos(rpy[0] * 0.5)
    sr = np.sin(rpy[0] * 0.5)
    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy
    q = Quaternion(q_w, q_x, q_y, q_z)
    return q


def quat_wxyz_to_xyzw(q):
    """
    Convert a quaternion from (w, x, y, z) to (x, y, z, w) format.
    """
    return np.array([q[1], q[2], q[3], q[0]])


def rot_coord(axis:str, theta:float) -> np.ndarray:
    """
    Compute rotation matrix for coordinate transformation. Note that
    coordinateRotation(CoordinateAxis:X, .1) * v will rotate v by 1 radians.
    """
    s = np.sin(float(theta))
    c = np.cos(float(theta))
    R:np.ndarray = None
    if axis == "x":
        R = np.array([1, 0, 0, 0, c, s, 0, -s, c], dtype=DTYPE).reshape((3, 3))
    elif axis == "y":
        R = np.array([c, 0, -s, 0, 1, 0, s, 0, c], dtype=DTYPE).reshape((3, 3))
    elif axis == "z":
        R = np.array([c, s, 0, -s, c, 0, 0, 0, 1], dtype=DTYPE).reshape((3, 3))

    return R.T
