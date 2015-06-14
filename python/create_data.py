#blender imports
import bpy
import bmesh
import bpy_extras.object_utils
import mathutils as mu

#std imports
import numpy as np
import os
import math
import datetime
from glob import glob
import random
import time

CAMERA = 'Camera'

ARM_BONE_NAMES = ['deltoid.R', 'upper_arm.R', 'forearm.R', 'hand.R']
HAND_BONE_NAMES = ['thumb.01.R', 'thumb.02.R', 'thumb.03.R',
                   'palm_index.R', 'f_index.01.R', 'f_index.02.R', 'f_index.03.R',
                   'palm_middle.R', 'f_middle.01.R', 'f_middle.02.R', 'f_middle.03.R',
                   'palm_ring.R', 'f_ring.01.R', 'f_ring.02.R', 'f_ring.03.R',
                   'palm_pinky.R', 'f_pinky.01.R', 'f_pinky.02.R', 'f_pinky.03.R']


def getTs():
  now = datetime.datetime.now()
  ts = '%04d%02d%02d%02d%02d%02d' % (now.year, now.month, now.day,
                                     now.hour, now.minute, now.second)
  return ts


#------------------------------------------------------------------------------
def printSelectedVertexInfo(mesh_name):
  object_reference = bpy.data.objects[mesh_name]

  bm = bmesh.from_edit_mesh(object_reference.data)
  for i, vert in enumerate(bm.verts):
    if vert.select:
      print('[VERT] array idx: %d; vert idx: %d ' % (i, vert.index))

  for i, face in enumerate(bm.faces):
    if face.select:
      print('[FACE] array idx: %d; face idx: %d ' % (i, face.index))
      print('[FACE] normal', face.normal)
      print(object_reference.data.polygons[i].normal)


def getPoseQuat(bone_names):
  state = np.zeros((4, len(bone_names)))
  for idx, bone_name in enumerate(bone_names):
    bone = bpy.context.object.pose.bones[bone_name]
    bone.rotation_mode = 'QUATERNION'
    for row in range(4):
      state[row, idx] = bone.rotation_quaternion[row]

  return state

def getArmPoseQuat():
  return getPoseQuat(ARM_BONE_NAMES)

def getHandPoseQuat():
  return getPoseQuat(HAND_BONE_NAMES)

def setPoseQuat(bone_names, state):
  for idx, bone_name in enumerate(bone_names):
    bone = bpy.context.object.pose.bones[bone_name]
    bone.rotation_mode = 'QUATERNION'
    for row in range(4):
      bone.rotation_quaternion[row] = state[row, idx]

def setArmPoseQuat(state):
  setPoseQuat(ARM_BONE_NAMES, state)

def setHandPoseQuat(state):
  setPoseQuat(HAND_BONE_NAMES, state)

def clearArmPoseQuat():
  state = np.zeros((4, len(ARM_BONE_NAMES)))
  state[0, :] = 1
  setPoseQuat(ARM_BONE_NAMES, state)

def clearHandPoseQuat():
  state = np.zeros((4, len(HAND_BONE_NAMES)))
  state[0, :] = 1
  setPoseQuat(HAND_BONE_NAMES, state)

def clearAllPoseQuat():
  clearArmPoseQuat()
  clearHandPoseQuat()


def writeArmPoseQuat(pose_dir):
  if not os.path.exists(pose_dir):
    os.makedirs(pose_dir)

  pose = getArmPoseQuat()

  ts = getTs()
  out_path = os.path.join(pose_dir, 'arm_pose_%s.csv' % ts)

  np.savetxt(out_path, pose, delimiter=',')

def writeHandPoseQuat(pose_dir):
  if not os.path.exists(pose_dir):
    os.makedirs(pose_dir)

  pose = getHandPoseQuat()

  ts = getTs()
  out_path = os.path.join(pose_dir, 'hand_pose_%s.csv' % ts)

  np.savetxt(out_path, pose, delimiter=',')


def setScale(scale):
  names = list(HAND_BONE_NAMES)
  names.extend(ARM_BONE_NAMES)

  for col, name in enumerate(names):
    for row in range(3):
      bpy.context.object.pose.bones[name].scale[row] = scale[row, col]


def writeScale(pose_dir):
  if not os.path.exists(pose_dir):
    os.makedirs(pose_dir)

  names = list(HAND_BONE_NAMES)
  names.extend(ARM_BONE_NAMES)

  scale = np.zeros((3, len(names)))
  for col, name in enumerate(names):
    for row in range(3):
      scale[row, col] = bpy.context.object.pose.bones[name].scale[row]

  ts = getTs()
  out_path = os.path.join(pose_dir, 'scale_%s.csv' % ts)

  print(out_path)
  print(scale)
  np.savetxt(out_path, scale, delimiter=',')


def readPose(pose_path):
  return np.genfromtxt(pose_path, delimiter=',')


def loadArmPose(path):
  pose = readPose(path)
  setArmPoseQuat(pose)

def loadHandPose(path):
  pose = readPose(path)
  setHandPoseQuat(pose)

def loadScale(path):
  scale = readPose(path)
  setScale(scale)

#------------------------------------------------------------------------------
def getJointAnno3d(mesh, mesh_vert_idx):
  wco = mesh.vertices[mesh_vert_idx].co
  cco = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, bpy.data.objects[CAMERA], wco)

  cco[0] = cco[0] * bpy.context.scene.render.resolution_x
  cco[1] = (1-cco[1]) * bpy.context.scene.render.resolution_y

  coord = np.ndarray((3,))
  coord[0] = cco[0]
  coord[1] = cco[1]
  coord[2] = cco[2]

  return coord

def getJointAnno2d(mesh_name, mesh, mesh_vert_idx):
  wco = mesh.vertices[mesh_vert_idx].co
  cco = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, bpy.data.objects[CAMERA], wco)

  cam = bpy.data.objects[CAMERA]
  scene = bpy.context.scene
  obj = bpy.data.objects[mesh_name]
  model_view = (
    cam.matrix_world.inverted() *
    obj.matrix_world
  )

  width =  scene.render.resolution_x
  height = scene.render.resolution_y
  aspect_ratio = width / height

  n = cam.data.clip_start
  f = cam.data.clip_end
  fov = cam.data.angle

  proj = mu.Matrix()
  proj[0][0] =            1 / math.tan(fov / 2)
  proj[1][1] = aspect_ratio / math.tan(fov / 2)
  proj[2][2] = -(f + n) / (f - n)
  proj[2][3] = - 2*f*n  / (f - n)
  proj[3][2] = - 1
  proj[3][3] =   0

  clip = proj * model_view

  v_4d = wco.copy()
  v_4d.resize_4d()

  v_clip  = clip * v_4d
  v_clip /= v_clip[3]
  v_co = v_clip.resized(3)

  scrn_co_x = (v_co.x + 1) / 2 * width
  scrn_co_y = (v_co.y + 1) / 2 * height

  coord = np.ndarray((3,))
  coord[0] = scrn_co_x
  coord[1] = height - scrn_co_y
  coord[2] = cco[2]

  return coord


def getAnno(mesh_name):
  bpy.data.scenes['Scene'].update()

  #deactivate modifiers except armature
  viewport_states = []
  for mod in bpy.data.objects[mesh_name].modifiers:
    viewport_states.append(mod.show_viewport)
    if mod.type != 'ARMATURE':
      mod.show_viewport = False

  #retrive posed mesh
  me = bpy.data.objects[mesh_name].to_mesh(scene=bpy.context.scene, apply_modifiers=True, settings='PREVIEW')

  #activiate modifiers except armature
  for i, mod in enumerate(bpy.data.objects[mesh_name].modifiers):
    mod.show_viewport = viewport_states[i]

  #joint coordinates
  anno3d = np.zeros((3, 20))
  vertice_idxs = [(3650, 3247),
                  (3603, 3639), (2713, 2720), (2677, 2777),
                  (1945, 3104), (1974, 1970), (1985, 1984), (1953, 2980),
                  (3269, 3110), (2194, 2095), (2177, 2176), (2145, 3019),
                  (2329, 3116), (2358, 2354), (2369, 2368), (2311, 2463),
                  (2521, 3124), (2637, 2478), (2636, 2597), (2529, 3097)]
  for joint_idx, vertice_idx in enumerate(vertice_idxs):
    vertice_anno = []
    if hasattr(vertice_idx, '__iter__'):
      for v_idx in vertice_idx:
        vertice_anno.append(getJointAnno3d(me, v_idx))
    else:
      vertice_anno.append(getJointAnno3d(me, vertice_idx))
    vertice_anno = np.array(vertice_anno)

    #print(80*'-')
    #print(vertice_anno)
    #print(np.mean(vertice_anno, axis=0))
    anno3d[:, joint_idx] = np.mean(vertice_anno, axis=0)

  #remove posed mesh
  bpy.data.meshes.remove(me)

  return anno3d


def render(depth_path, rgb_path, render_img):
  bpy.data.scenes['Scene'].update()

  bpy.context.scene.node_tree.nodes['FO_depth'].base_path = os.path.dirname(depth_path)
  bpy.context.scene.node_tree.nodes['FO_depth'].file_slots[0].path = os.path.basename(depth_path)

  if render_img:
    bpy.context.scene.node_tree.nodes['FO_rgb'].base_path = os.path.dirname(rgb_path)
    bpy.context.scene.node_tree.nodes['FO_rgb'].file_slots[0].path = os.path.basename(rgb_path)

  bpy.ops.render.render()


#def getIntrinsics():
  ## http://cmp.felk.cvut.cz/ftp/articles/svoboda/Mazany-TR-2007-02.pdf
  #f = bpy.data.cameras["Camera"].lens / 16.0
  #width = bpy.context.scene.render.resolution_x
  #height = bpy.context.scene.render.resolution_y
  #kv = bpy.context.scene.render.pixel_aspect_x
  #ku = bpy.context.scene.render.pixel_aspect_y

  #if(width * kv > height * ku):
    #mv = width / 2
    #mu = mv * kv / ku
  #else:
    #mu = height / 2
    #mv = mu * ku / kv

  #f_x = mu * f
  #f_y = mv * f
  #p_x = width / 2
  #p_y = height / 2

  #return (f_x, f_y, p_x, p_y)


def getIntrinsics():
  # http://kwunlyou.com/blog/graphics/2012/03/17/simulate-real-cameras-in-blender.html
  scn = bpy.data.scenes['Scene']

  size_w = scn.render.resolution_x
  size_h = scn.render.resolution_y
  per = scn.render.resolution_percentage
  size_w = size_w*per/100
  size_h = size_h*per/100

  cam1 = bpy.data.cameras["Camera"]

  lens = cam1.lens
  sensor_w = cam1.sensor_width
  sensor_h = cam1.sensor_height

  #matlab
  # pixel to unit
  sensor_w = sensor_w / size_w;
  sensor_h = sensor_h / size_h;

  # shift
  shift_w = size_w / 2 * sensor_w;
  shift_h = size_h / 2 * sensor_h;

  # compute K
  f_x = lens / sensor_w
  f_y = lens / sensor_h
  p_x = shift_w / sensor_w
  p_y = shift_h / sensor_h

  return (f_x, f_y, p_x, p_y)

def writeIntrinsics(path):
  f_x, f_y, p_x, p_y = getIntrinsics()

  with open(path, 'w') as f:
    f.write("%f %f\n" % (f_x, f_y))
    f.write("%f %f\n" % (p_x, p_y))
    f.write("0.0 0.0 0.0 0.0 0.0\n")
    f.close()

def loadPoses(pose_paths):
  for idx, pose_path in enumerate(pose_paths):
    if idx == 0:
      pose = readPose(pose_path)
      poses = np.ndarray((len(pose_paths), pose.shape[0], pose.shape[1]))
      poses[0, :, :] = pose
    else:
      poses[idx, :, :] = readPose(pose_path)

  return poses


def setScaleHand(x, y, z, scale_palm):
  bones = list(HAND_BONE_NAMES)
  if scale_palm:
    bones.append(ARM_BONE_NAMES[-1])

  for name in bones:
    bpy.context.object.pose.bones[name].scale[0] = x
    bpy.context.object.pose.bones[name].scale[1] = y
    bpy.context.object.pose.bones[name].scale[2] = z


def sequentialIdxPairs(n): # idx with idx+1
  return [(idx, idx+1) for idx in range(n-1)]

def allIdxPairs(n): #every idx with all other
  return [(idx1, idx2) for idx1 in range(n-1) for idx2 in range(ix1+1, n)]

def randKIdxPairs(n, k): # idx with random k others
  idx_pairs = []
  for idx_1 in range(n):
    for k_ in range(k):
      idx_2 = random.randint(0, n-1)
      while idx_2 == idx_1:
        idx_2 = random.randint(0, n-1)
      idx_pairs.append((idx_1, idx_2))
  return idx_pairs

def rand1IdxPairs(n):
  return randKIdxPairs(n, 1)

def rand2IdxPairs(n):
  return randKIdxPairs(n, 2)

def rand3IdxPairs(n):
  return randKIdxPairs(n, 3)

def interpolatePoses(poses, steps, createIdxPairs):
  if poses.shape[0] == 1:
    return poses.copy()

  idx_pairs = createIdxPairs(poses.shape[0])

  space = np.linspace(0.0, 1.0, steps)

  inter_idx = 0
  interpolations = np.zeros((len(idx_pairs) * steps, poses.shape[1], poses.shape[2]))
  for idx_1, idx_2 in idx_pairs:
    for s in space:
      interpolations[inter_idx] = s * poses[idx_1] + (1.0 - s) * poses[idx_2]
      inter_idx += 1

  return interpolations


def generate_(mesh_name, data_dir, pose_dir, arm_interpolation_steps, hand_interpolation_steps, n_jitter, render_img):
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  scale_paths = glob(os.path.join(pose_dir, 'scale*csv'))
  arm_pose_paths = glob(os.path.join(pose_dir, 'arm_pose*csv'))
  hand_pose_paths = glob(os.path.join(pose_dir, 'hand_pose*csv'))

  print("%d sclae_paths" % len(scale_paths))
  print("%d arm_pose_paths" % len(arm_pose_paths))
  print("%d hand_pose_paths" % len(hand_pose_paths))

  scales = loadPoses(scale_paths)
  arm_poses = loadPoses(arm_pose_paths)
  hand_poses = loadPoses(hand_pose_paths)

  arm_poses = interpolatePoses(arm_poses, arm_interpolation_steps, rand1IdxPairs)
  hand_poses = interpolatePoses(hand_poses, hand_interpolation_steps, rand2IdxPairs);

  #write camera parameters
  calibration_camera_path = os.path.join(data_dir, 'calibration_depth.txt')
  writeIntrinsics(calibration_camera_path)

  start_time = time.time()

  #render each hand_pose at every arm_pose
  img_idx = 0
  n_images = scales.shape[0] * arm_poses.shape[0] * hand_poses.shape[0] * (1 + n_jitter)
  for scale in scales:
    setScale(scale)

    for arm_pose in arm_poses:
      setArmPoseQuat(arm_pose)

      for clean_hand_pose in hand_poses:

        for jitter_idx in range(1 + n_jitter):
          print('render %d of %d' % (img_idx + 1, n_images))

          # jitter if idx > 0
          if jitter_idx == 0:
            hand_pose = clean_hand_pose
            out_dir = os.path.join(data_dir, 'clean')
          else:
            jitter = np.zeros((4, len(HAND_BONE_NAMES)))
            jitter[1, :] = np.random.rand(1, jitter.shape[1]) * 0.15 - 0.05 # [-0.05, 0.15]
            jitter[2, :] = np.random.rand(1, jitter.shape[1]) * 0.1 - 0.05 # [-0.05, 0.05]
            jitter[3, :] = np.random.rand(1, jitter.shape[1]) * 0.1 - 0.05 # [-0.05, 0.05]
            hand_pose = clean_hand_pose + jitter

            out_dir = os.path.join(data_dir, 'jitter_%d' % jitter_idx)

          if not os.path.exists(out_dir):
            os.makedirs(out_dir)

          setHandPoseQuat(hand_pose)

          # define output paths
          ts = getTs()
          depth_path = os.path.join(out_dir, '%08d_%s_depth_' % (img_idx, ts))
          rgb_path = os.path.join(out_dir, '%08d_%s_rgb_' % (img_idx, ts))
          anno3d_path = os.path.join(out_dir, '%08d_%s_anno_blender.txt' % (img_idx, ts))

          #create imgs
          render(depth_path, rgb_path, render_img)

          #compute anno
          anno3d = getAnno(mesh_name)
          anno3d[2, :] *= 1000 # to mm

          #save anno
          anno3d = anno3d.T.reshape(1, np.prod(anno3d.shape))
          np.savetxt(anno3d_path, anno3d, delimiter=' ', fmt='%.4f')

          #save anno constraint for first img
          if img_idx == 0:
            np.savetxt(os.path.join(out_dir, 'anno_constraint.txt'), anno3d, delimiter=' ', fmt='%.4f')

          img_idx += 1

          #fps / remaining time
          fps = float(img_idx) / (time.time() - start_time)

          n_images_left = n_images - img_idx
          remaining_time_min = n_images_left / fps / 60
          remaining_time_h = math.floor(remaining_time_min / 60)
          remaining_time_min = remaining_time_min - remaining_time_h * 60
          print("fps = %f | remaining: %d:%f" % (fps, remaining_time_h, remaining_time_min))


def generate(data_dir, pose_dir, mhx_dir, arm_interpolation_steps, hand_interpolation_steps, n_jitter, render_img):
  mesh_name = "m_25:Body"

  #define what to render
  print("render_img: %d" % render_img)
  bpy.context.scene.render.layers["RenderLayer"].use_pass_combined = render_img
  bpy.context.scene.render.layers["RenderLayer"].use_pass_z = True


#  print(mhx_dir)
#  for mhx_path in glob(os.path.join(mhx_dir, '*mhx')):
#    print(mhx_path)
#
#    #delete armature and mesh from blender
#    bpy.ops.object.mode_set(mode='OBJECT')
#    for ob in bpy.context.scene.objects:
#        ob.select = ob.type == "ARMATURE" or ob.type == "MESH"
#    bpy.ops.object.delete()
#
#    #import scene
#    bpy.ops.import_scene.makehuman_mhx(filepath=mhx_path)
#
#    file_name = os.path.splitext(os.path.basename(mhx_path))[0]
#    mesh_name = file_name + ":Body"
#    out_dir = os.path.join(data_dir, file_name)
#    generate_(mesh_name, out_dir, pose_dir, arm_interpolation_steps, hand_interpolation_steps)


  out_dir = os.path.join(data_dir, "generic")
  generate_(mesh_name, out_dir, pose_dir, arm_interpolation_steps, hand_interpolation_steps, n_jitter, render_img)

  print("finished")


