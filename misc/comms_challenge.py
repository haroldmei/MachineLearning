"""
######
#.#E.#
#....#
######

# : wall
. : floor
E : Exit

U, D, L, R

UDUDLRLR

Input: data of Maze (list of #, ., E)
Output: List of commands

Output: DRRUL

######
#1#E2#
#3456#
######

######
# #E #
#   S#
######

######
#o#Ex#
#o/xx#
######

######
#1#E/#
#3///#
######

get_plan(maze, pos1) -> U, L

U, D, R, L 


R, U: 


"""

def get_command(cur, parent):
  if cur[0] - 1 == parent[0]:
    return 'R'
  if cur[0] + 1 == parent[0]:
    return 'L'
  if cur[1] - 1 == parent[1]:
    return 'D'
  if cur[1] + 1 == parent[1]:
    return 'U'
  

def plan(maze): # mxn, 8 by 8 
  E = [0,0]
  for i in range(len(maze[0])):
    for j in range(len(maze)):
      if maze[j][i] == 'E':
        E = (i,j)
        
  fifo = deque([E])
  visited = set([])
  commands = {}
  
  while fifo:
    x,y = fifo.popleft()
    pos1 = [x - 1, y]
    pos2 = [x + 1, y]
    pos3 = [x, y + 1]
    pos4 = [x, y - 1]
    for p in [pos1,pos2,pos3,pos4]:
      if maze[p[0]][p[1]] != '#' (p[0], p[1]) not in visited;
        #parents[(p[0], p[1])] = (x,y)
        
        cmd = get_command((p[0], p[1]), (x,y))
        commands[((p[0], p[1]))] = cmd
        
        fifo.append((p[0], p[1]))
  return commands


def get_next(cur, command):
  if command == 'R':
    cur[0] += 1
  if command == 'L':
    cur[0] -= 1
  if command == 'U':
    cur[1] -= 1
  if command == 'D':
    cur[1] += 1
  return cur


def get_plan(maze, cur):
  commands = plan(maze)
  res = []
  while True:
    cmd = commands[cur]
    cur = get_next(cur, cmd)
    res.append(cmd)
    
  return res

def getAllPlans(maze):
  
  for i in range(len(maze[0])):
    for j in range(len(maze)):
      if maze[j][i] == '.':
        plan = get_plan(maze, (i,j))
        res = get_route(maze, plan, (i,j))
        if res:
          return res
  return []


def get_route(maze, plan, cur):
  res = []
  for cmd in plan:
    next = get_next(cur, cmd)
    if next != cur:
      res.append(cmd) 
    else:
      return []
  return res
  

