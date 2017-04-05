import simpy
from random import randint

class EV:
    def __init__(self, env):
        self.env = env
        self.drive_proc = env.process(self.drive(env))

    def drive(self, env):
        while True:
            # Drive for 20-40 min
            yield env.timeout(randint(20, 40))

            # Park for 1 hour
            print('Start parking at', env.now)
            charging = env.process(self.bat_ctrl(env))
            parking = env.timeout(60)
            yield charging | parking
            if not charging.triggered:
                # Interrupt charging if not already done.
                charging.interrupt('Need to go!')
            print('Stop parking at', env.now)

    def bat_ctrl(self, env):
        print('Bat. ctrl. started at', env.now)
        try:
            yield env.timeout(randint(60, 90))
            print('Bat. ctrl. done at', env.now)
        except simpy.Interrupt as i:
            # Onoes! Got interrupted before the charging was done.
            print('Bat. ctrl. interrupted at', env.now, 'msg:',
                  i.cause)

def proc_two(env):
	print('Process 2 Yielding...')
	try:
		yield env.timeout(50)
	except simpy.Interrupt as i:
		print('Process 2 Interrupted')


def proc_one(env):
	proc2 = env.process(proc_two(env))
	print('Process 1 Yielding...')
	yield env.timeout(30)
	print('Process 1 Yielded')
	if not proc2.triggered:
		print('Interrupting Process 2')
		proc2.interrupt()



env = simpy.Environment()
# ev = EV(env)
# env.run(until=100)

env.process(proc_one(env))

import pdb
pdb.set_trace()

# time = 0.

# for i in range(20):
# 	while(env.peek() <= time):
# 		print('		.')
# 		env.step()
# 	time = env.peek()
# 	print('Stepping...')
# 	print('Time: ', env.now)
# 	env.step()
