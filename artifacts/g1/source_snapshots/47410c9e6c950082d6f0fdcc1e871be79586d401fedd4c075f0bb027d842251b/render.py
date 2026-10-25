"""Pillow renderer driven only by actual recorded world states."""
import numpy as np
from PIL import Image, ImageDraw

COLORS = ('#2474c6', '#209657', '#a445b1', '#db8b1d')


def frame(world):
    image = Image.new('RGB', (640, 720), '#f5f6fa')
    draw = ImageDraw.Draw(image)
    scale = 60
    def xy(p):
        return (20 + p[0] * scale, 620 - p[1] * scale)
    draw.rectangle((20, 20, 620, 620), fill='white', outline='#303544', width=2)
    for obstacle in world.scene['obstacles']:
        if obstacle['kind'] == 'circle':
            x, y = xy(obstacle['center'])
            r = obstacle['radius'] * scale
            draw.ellipse((x-r, y-r, x+r, y+r), fill='#929aaa')
        else:
            box = obstacle['box']
            x1, y1 = xy(box[:2])
            x2, y2 = xy(box[2:])
            draw.rectangle((x1, y2, x2, y1), fill='#929aaa')
    for i, state in enumerate(world.state):
        color = COLORS[i % len(COLORS)]
        gx, gy = xy(world.goals[i])
        gr = world.cfg['goal_tolerance_m'] * scale
        draw.ellipse((gx-gr, gy-gr, gx+gr, gy+gr), outline=color, width=3)
        draw.text((gx-4, gy-5), str(i+1), fill=color)
        x, y = xy(state[:2])
        r = world.cfg['robot_radius_m'] * scale
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline='black')
        draw.line((x, y, x+r*np.cos(state[2]), y-r*np.sin(state[2])), fill='white', width=3)
    status = 'COLLISION' if world.contact else 'SUCCESS' if world.finished.all() else 'TIMEOUT' if world.truncated else 'RUNNING'
    draw.text((20, 638), f"RoboRace G1 | classical | {world.scene['course']} | seed {world.scene['seed']}", fill='#202634')
    draw.text((20, 658), f"Simulated time {world.time:.1f}s | arrived {sum(world.finished)}/{world.n} | {status}", fill='#202634')
    draw.text((20, 680), '2D simulation | 1x playback | circles = robots; rings = goals', fill='#202634')
    return image
