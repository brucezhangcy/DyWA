#!/usr/bin/env python3
from xml.dom import minidom

with open('ddg-gd_pitcher_poisson_005-0.060.urdf', 'r') as fp:
    str_urdf = fp.read()
dom = minidom.parseString(str_urdf)
meshes = dom.getElementsByTagName("mesh")
scale:float=0.2
for mesh in meshes:
    mesh_scales = mesh.attributes['scale'].value.split(' ')
    new_scale = [str(scale* float(mesh_scale))
                    for mesh_scale in mesh_scales]
    mesh.attributes['scale'].value = (
        ' '.join(new_scale)
    )
    origin = mesh.parentNode.parentNode.getElementsByTagName('origin')
    for o in origin:
        old_xyz = o.attributes['xyz'].value.split(' ')
        new_xyz = [str(scale* float(oo)) for oo in old_xyz]
        o.attributes['xyz'].value = (
                ' '.join(new_xyz)
        )
