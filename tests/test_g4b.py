import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('pybullet')
from roborace.bullet_world import BulletWorld
from roborace.scenarios import generate

CFG=json.loads((Path(__file__).parents[1]/'configs/g0_contract.json').read_text())


def test_reset_motion_sensor_and_determinism():
    w=BulletWorld(CFG)
    try:
        scene=generate(41000)
        a=w.reset(scene=scene);initial=w.state.copy()
        assert a.shape==(3,44)
        assert np.all(a[:,43]==0) # absent third peer
        obs,*_=w.step(np.array([[.5,0]]*3))
        pose=w.state.copy()
        np.testing.assert_allclose(pose[:,:2]-initial[:,:2],np.column_stack((.05*np.cos(initial[:,2]),.05*np.sin(initial[:,2]))),atol=1e-5)
        w.reset(scene=scene);w.step(np.array([[.5,0]]*3))
        np.testing.assert_allclose(w.state,pose,atol=1e-8)
        w.reset(scene=scene);w.step(np.array([[0,1.]]*3))
        np.testing.assert_allclose(w.state[:,2]-initial[:,2],.1,atol=1e-5)
    finally:w.close()


@pytest.mark.parametrize('peer',[False,True])
def test_physics_contacts(peer):
    w=BulletWorld(CFG)
    try:
        s=generate(41000)
        s['starts']=[[1,2,0],[1.65,2,0] if peer else [1,5,0],[1,8,0]]
        if not peer:s['obstacles']=[{'kind':'circle','center':[1.7,2],'radius':.2}]
        w.reset(scene=s)
        for _ in range(20):
            w.step(np.array([[1,0],[0,0],[0,0]]))
            if w.terminated:break
        assert w.contact and w.contact['type']==('peer' if peer else 'static')
    finally:w.close()


def test_arrival_stop():
    w=BulletWorld(CFG)
    try:
        s=generate(41000);s['goals']=[r[:2] for r in s['starts']]
        w.reset(scene=s)
        for _ in range(CFG['arrival_dwell_steps']):w.step(np.zeros((3,2)))
        assert w.summary()['team_success']
    finally:w.close()


def test_forward_range_matches_known_obstacle():
    w=BulletWorld(CFG)
    try:
        s=generate(41000);s['starts'][0]=[1,2,0]
        s['obstacles']=[{'kind':'circle','center':[3,2],'radius':.4}]
        obs=w.reset(scene=s)
        assert obs[0,5]*5 == pytest.approx(1.6,abs=1e-6)
        assert obs[0,33]==1 and obs[0,38]==1 and obs[0,43]==0
    finally:w.close()
