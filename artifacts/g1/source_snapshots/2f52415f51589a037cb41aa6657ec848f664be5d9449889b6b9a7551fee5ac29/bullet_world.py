"""CPU PyBullet velocity-servo robots; no trajectory replay or fallback policy."""
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw
from .env import World


class BulletWorld(World):
    collision_retains_actual_pose = True
    def __init__(self, config):
        super().__init__(config)
        self.client = p.connect(p.DIRECT)

    def close(self):
        p.disconnect(self.client)

    def reset(self, seed=0, course='open', scene=None):
        super().reset(seed, course, scene)
        c = self.client
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, 0, physicsClientId=c)
        p.setTimeStep(self.cfg['control_dt_s']/self.cfg['physics_substeps'], physicsClientId=c)
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=c)
        self.robots, self.static = [], []
        def body(shape, position, color, mass=0, **kw):
            col = p.createCollisionShape(shape, physicsClientId=c, **kw)
            vkw = dict(kw)
            if 'height' in vkw: vkw['length'] = vkw.pop('height')
            vis = p.createVisualShape(shape, rgbaColor=color, physicsClientId=c, **vkw)
            return p.createMultiBody(mass, col, vis, position, physicsClientId=c)
        body(p.GEOM_BOX, [5,5,-.08], [.14,.19,.24,1], halfExtents=[5,5,.08])
        for pos, half in [([-0.1,5,.3],[.1,5,.3]),([10.1,5,.3],[.1,5,.3]),([5,-.1,.3],[5,.1,.3]),([5,10.1,.3],[5,.1,.3])]:
            self.static.append(body(p.GEOM_BOX,pos,[.4,.47,.55,1],halfExtents=half))
        for ob in self.scene['obstacles']:
            if ob['kind']=='circle':
                self.static.append(body(p.GEOM_CYLINDER,[*ob['center'],.3],[.95,.55,.16,1],radius=ob['radius'],height=.6))
            else:
                x,y,X,Y=ob['box']
                self.static.append(body(p.GEOM_BOX,[(x+X)/2,(y+Y)/2,.3],[.95,.55,.16,1],halfExtents=[(X-x)/2,(Y-y)/2,.3]))
        colors=[[.2,.65,1,1],[.3,.9,.55,1],[.85,.4,.95,1]]
        self.decor=[]
        for i,(x,y,yaw) in enumerate(self.state):
            r=body(p.GEOM_CYLINDER,[x,y,.18],colors[i],mass=1,radius=self.cfg['robot_radius_m'],height=.24)
            p.resetBasePositionAndOrientation(r,[x,y,.18],p.getQuaternionFromEuler([0,0,yaw]),physicsClientId=c)
            p.changeDynamics(r,-1,linearDamping=0,angularDamping=0,physicsClientId=c)
            self.robots.append(r)
            # Visual wheel/heading markers follow measured robot poses, never drive them.
            for dx,dy in [(0,.18),(0,-.18),(.17,0)]:
                vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[.09,.035,.065],rgbaColor=[.04,.05,.07,1] if dx==0 else [1,1,1,1],physicsClientId=c)
                self.decor.append((p.createMultiBody(0,-1,vis,[x+dx,y+dy,.15],physicsClientId=c),i,dx,dy))
            for point,color in [(self.scene['starts'][i][:2],[.3,.4,.5,1]),(self.goals[i],colors[i])]:
                vis=p.createVisualShape(p.GEOM_CYLINDER,radius=.32,length=.012,rgbaColor=color,physicsClientId=c)
                p.createMultiBody(0,-1,vis,[*point,.012],physicsClientId=c)
        return self.observe()

    def advance(self, old, actions, dt):
        c=self.client
        for i,r in enumerate(self.robots):
            yaw=old[i,2]
            p.resetBaseVelocity(r,[actions[i,0]*np.cos(yaw),actions[i,0]*np.sin(yaw),0],[0,0,actions[i,1]],physicsClientId=c)
        p.stepSimulation(physicsClientId=c)
        new=old.copy()
        contact=None
        for i,r in enumerate(self.robots):
            pos,quat=p.getBasePositionAndOrientation(r,physicsClientId=c)
            new[i]=[*pos[:2],p.getEulerFromQuaternion(quat)[2]]
            v,w=p.getBaseVelocity(r,physicsClientId=c)
            self.velocity[i]=[v[0]*np.cos(new[i,2])+v[1]*np.sin(new[i,2]),w[2]]
            for hit in p.getContactPoints(bodyA=r,physicsClientId=c):
                other=hit[2]
                if other in self.static or other in self.robots:
                    contact={'type':'peer' if other in self.robots else 'static','robot':i,'body':other,'distance_m':hit[8]}
        return new,contact

    def camera(self, label):
        c=self.client
        for b,i,dx,dy in self.decor:
            x,y,yaw=self.state[i]
            p.resetBasePositionAndOrientation(b,[x+dx*np.cos(yaw)-dy*np.sin(yaw),y+dx*np.sin(yaw)+dy*np.cos(yaw),.19],p.getQuaternionFromEuler([0,0,yaw]),physicsClientId=c)
        view=p.computeViewMatrix([13,-9,15],[5,5,0],[0,0,1])
        proj=p.computeProjectionMatrixFOV(48,960/720,.1,60)
        rgba=p.getCameraImage(960,720,view,proj,renderer=p.ER_TINY_RENDERER,physicsClientId=c)[2]
        im=Image.fromarray(np.asarray(rgba,dtype=np.uint8).reshape(720,960,4)[:,:,:3])
        d=ImageDraw.Draw(im)
        d.rectangle((0,0,960,60),fill='#101923')
        d.text((16,10),f'RoboRace 3D | {label} | {self.time:.1f}s | 1x playback',fill='white')
        d.text((16,32),f'R1 blue / R2 green / R3 purple | finishes: {self.finish_times} | contact: {bool(self.contact)}',fill='white')
        return im
