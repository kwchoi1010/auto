# - 선박에 Lidar 추가, 장애물 확인 시 거리계산, 점 표현
# - sub2에 Lidar 값 plot
# - Lidar기반 충돌회피 알고리즘

import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from drawnow import *
import matplotlib.patches as patches

def ssa(angle, unit='rad'):
    if unit == 'deg':
            angle = ((angle + 180) % 360) - 180
    else:
        angle = ((angle + math.pi) % (2 * math.pi)) - math.pi
    return angle

def sat(x_sat, xmax):
    if abs(x_sat) >= xmax:
        y_sat = np.sign(x_sat) * xmax
    else:
        y_sat = x_sat
    return y_sat

def vehShape(stParam, veh):
    vehx = stParam['veh_info']['L'] * np.array([0.5, 1, 0.5, -1, -1, 0.5])
    vehy = stParam['veh_info']['B'] * np.array([1, 0, -1, -1, 1, 1])
    veh_x = veh[0]
    veh_y = veh[1]
    veh_psi = veh[2]
    veh_spd = veh[3]
    
    posx = veh_x + vehx*np.cos(veh_psi) - vehy*np.sin(veh_psi)
    posy = veh_y + vehx*np.sin(veh_psi) + vehy*np.cos(veh_psi)

    h = []
    h.append(sub1.plot(veh_x, veh_y, 'ro', linewidth=1)[0])   # center
    h.append(sub1.plot(posx, posy, 'r-', linewidth=1)[0])     # shape
    h.append(sub1.plot([veh_x, veh_x + (stParam['veh_info']['headDir'] * veh_spd) * np.cos(veh_psi)],
                      [veh_y, veh_y + (stParam['veh_info']['headDir'] * veh_spd) * np.sin(veh_psi)],
                      'r--', linewidth=1)[0]) # heading line depending on speed
    h.extend(plot2Dssz(stParam, veh, 'c'))
    return h

def plot2Dssz(stParam, veh, color):
    ssz_r = stParam['veh_info']['ssz']
    theta = np.arange(0, 361) / 180 * math.pi
    x_ssz = ssz_r * np.cos(theta) + veh[0]
    y_ssz = ssz_r * np.sin(theta) + veh[1]
    
    h = []
    h.append(sub1.plot(x_ssz, y_ssz, color, linewidth=1)[0])
    h.append(sub1.plot(veh[0], veh[1], 'rs', linewidth=1)[0])    # center
    return h

def shipModel(pos_x, pos_y, course_angle, speed):
    stParam = {}
    stParam['veh_info'] = {}
    stParam['veh_info']['L'] = 1.5
    stParam['veh_info']['B'] = 1.5 / 2
    stParam['veh_info']['buffer'] = 2.5 * stParam['veh_info']['L']
    stParam['veh_info']['headDir'] = 5
    stParam['veh_info']['ssz'] = 10        # buffer = safe separation zone (ssz)

    stParam['dt'] = 0.1
    stParam['tf'] = 10

    veh = np.array([pos_x, pos_y, course_angle, speed])    # pos_x, pos_y, heading (rad), speed (m/s)
    h = vehShape(stParam, veh)

    lidar_angles = np.linspace(-np.pi/2, np.pi/2, 36)
    lidar_range = 50
    lidar_pos_x = pos_x - stParam['veh_info']['L'] * np.sin(course_angle)
    lidar_pos_y = pos_y + stParam['veh_info']['L'] * np.cos(course_angle)
    for angle in lidar_angles:
        dx = lidar_range * np.cos(angle + course_angle)
        dy = lidar_range * np.sin(angle + course_angle)
        x1 = lidar_pos_x
        y1 = lidar_pos_y
        x2 = lidar_pos_x + dx
        y2 = lidar_pos_y + dy
        h.append(sub1.plot([x1, x2], [y1, y2], 'b-', linewidth=1)[0])

    return h

def LOSchi(x_los, y_los, Delta, R_switch, wpt_pos_x, wpt_pos_y, waypoint_index):
    
    global k
    global xk
    global yk
    
    # Initialization of (xk, yk) and (xk_next, yk_next)
    if 'k' not in globals():
        # check if R_switch is smaller than the minimum distance between the waypoints
        if R_switch > np.min(np.sqrt(np.diff(wpt_pos_x)**2 + np.diff(wpt_pos_y)**2)):
            raise ValueError("The distances between the waypoints must be larger than R_switch")
        
        # check input parameters
        if R_switch < 0:
            raise ValueError("R_switch must be larger than zero")
        if Delta < 0:
            raise ValueError("Delta must be larger than zero")
        
        k = 1  # set first waypoint as the active waypoint
        xk = wpt_pos_x[k-1]
        yk = wpt_pos_y[k-1]

    ## Read next waypoint (xk_next, yk_next) from wpt.pos 
    n = len(wpt_pos_x)
    if k < n:  # if there are more waypoints, read next one 
        xk_next = wpt_pos_x[k]
        yk_next = wpt_pos_y[k]
    else:  # else, use the last one in the array
        xk_next = wpt_pos_x[-1]
        yk_next = wpt_pos_y[-1]

    # Compute the desired course angle
    pi_p = math.atan2(yk_next-yk, xk_next-xk)  # path-tangential angle w.r.t. to North

    # along-track and cross-track errors (x_e, y_e) expressed in NED
    x_e =  (x_los-xk) * np.cos(pi_p) + (y_los-yk) * np.sin(pi_p)
    y_e = -(x_los-xk) * np.sin(pi_p) + (y_los-yk) * np.cos(pi_p)

    # Waypoint update
    x_error = -xk_next + x_los
    y_error = -yk_next + y_los
    R = np.sqrt(x_error**2 + y_error**2)
    if (R < R_switch) and (k < n):
        k = k + 1
        waypoint_index += 1
        xk = xk_next
        yk = yk_next
    # LOS guidance law
    Kp = 1/Delta
    chi_d = pi_p - np.arctan(Kp * y_e)

    return chi_d, y_e, waypoint_index

def WAMV_CNU(states, control):
    Tp = control[0]
    Ts = control[1]

    u_wamv = states[0]
    v_wamv = states[1]
    r_wamv = states[2]
    psi_wamv = states[5]

    xdot = np.array([-1.1391*u_wamv+0.0028*(Tp+Ts)+0.6836,
                     0.0161*v_wamv-0.0052*r_wamv+0.002*(Tp-Ts)*2.44/2+0.0068,
                     8.2861*v_wamv-0.9860*r_wamv+0.0307*(Tp-Ts)*2.44/2+1.3276,
                     u_wamv*np.cos(psi_wamv)-v_wamv*np.sin(psi_wamv),
                     u_wamv*np.sin(psi_wamv)+v_wamv*np.cos(psi_wamv),
                     r_wamv])
    
    return xdot


wpt_pos_x = []
wpt_pos_y = []

# figure 생성
plt.figure(figsize=(12,9))
sub1= plt.subplot(2,1,1)
sub1.set(xlim=[0.0, 600.0],
         ylim=[0.0, 300.0],
         title='Guidance + PID control map',
         xlabel='$x$-position [m]',
         ylabel='$y$-position [m]')

sub2 = plt.subplot(2,2,3, projection='polar')
sub3 = plt.subplot(2,2,4)
sub3.set(title = 'Thrust (N)',
         xlabel = 'time (s)',
         ylim=[-400, 400])
sub3.grid()
sub3.legend(['T_max', 'T_min', 'T_p', 'T_s'])
rect1 = patches.Rectangle((180, 0), 30, 180, linewidth=1, edgecolor='g', facecolor='g')
sub1.add_patch(rect1)

rect2 = patches.Rectangle((400, 120), 30, 180, linewidth=1, edgecolor='g', facecolor='g')
sub1.add_patch(rect2)

# rect = patches.Rectangle((200, 100), 100, 50, linewidth=1, edgecolor='r', facecolor='r')
# sub1.add_patch(rect)

# 클릭 이벤트 핸들러 함수
def onclick(event):
    global wpt_pos_x, wpt_pos_y
    
    # 마우스 왼쪽 버튼 클릭 시
    if event.button == 1:
        # 클릭한 점의 x, y 좌표를 wpt_pos_x, wpt_pos_y 리스트에 추가
        wpt_pos_x.append(event.xdata)
        wpt_pos_y.append(event.ydata)
        
        # 현재까지 입력된 waypoint들을 시각화
        sub1.plot(wpt_pos_x, wpt_pos_y, 'ro')
        
    # 마우스 우클릭 시
    elif event.button == 3:
        # 가장 최근에 추가된 점을 삭제
        if wpt_pos_x and wpt_pos_y:
            wpt_pos_x.pop()
            wpt_pos_y.pop()
            
            # 현재까지 입력된 waypoint들을 시각화
            sub1.cla()
            sub1.set(xlim=[0.0, 600.0],
                     ylim=[0.0, 300.0],
                     title='Guidance + PID control map',
                     xlabel='$x$-position [m]',
                     ylabel='$y$-position [m]')
            sub1.plot(wpt_pos_x, wpt_pos_y, 'ro')
    
    # 그래프 업데이트
    plt.draw()

def onkeypress(event):
    if event.key == ' ':     
        
        start_X = wpt_pos_x[0]
        start_Y = wpt_pos_y[0]
        goal_X = wpt_pos_x[-1]
        goal_Y = wpt_pos_y[-1]

        path_x = []
        path_y = []
        
        x_max = 500
        y_max = 500
        plt.axis([0, x_max, 0, y_max])

        # Own ship module _ waypoints
        h = 0.1
        N = 200000

        # initial values for x = [ u v r x y psi ]'
        x = np.zeros((6, 1))
        x[0] = 1
        x[5] = math.pi/4

        # PID course autopilot (Nomoto gains)
        T = 1
        m = 41.4  # m = T/K
        K = T / m
        wn = 1.5  # pole placement parameters
        zeta = 1
        Kp = m * wn**2
        Kd = m * (2 * zeta * wn - 1/T)
        Td = Kd/Kp
        Ti = 10/wn
        Ki = Kp*(1/Ti)

        # Reference model
        wn_d = 1.2  # natural frequency
        zeta_d = 1.0  # relative damping factor
        omega_d = 0
        a_d = 0

        # Propeller dynamics
        # Load condition
        mp = 25  # payload mass (kg), max value 45 kg
        rp = np.array([0, 0, -0.35]).reshape((3,1))

        # Current
        V_c = 1  # current speed (m/s)
        B = [[1, 1], [1.22, -1.22]]
        Binv = np.linalg.inv(B)

        # MAIN LOOP(Test)
        simdata = np.zeros((N+1, 25))                # table for simulation data
        R_switch = 4
        Delta = 50

        psi_init = math.atan2((wpt_pos_y[1]-wpt_pos_y[0]), (wpt_pos_x[1]-wpt_pos_x[0]))
        # Set ownship initial position at the first waypoint
        wp_index = 0
        x[3] = wpt_pos_x[wp_index]
        x[4] = wpt_pos_y[wp_index]
        x[5] = psi_init

        # xd = [0; 0; 0];
        xd = np.array([[psi_init], [0], [0]])

        chi_error_past = 0
        chi_error_sum = 0                       # integral state

        surge_error_past = 0    
        surge_error_sum = 0
        Kp_surge = 1500
        Kd_surge = 100
        Ki_surge = 100

        # surge reference speed
        U_ref = 2
        u_d_min = 1e-10
        x_hat = np.zeros((5,1))
        pre_waypoint_index = 1
        waypoint_index = 1
        num = len(wpt_pos_x)
        movie_iter = 0

        # axis([0 x_max 0 y_max]);
        # axis equal;
        T_max = 400
        T_min = -400

        for i in range(1,N+2):
            t = (i-1) * h  # time (s)
        
            u = x[0][0]
            v = x[1][0]
            U = math.sqrt(u**2 + v**2)  # speed
            psi = x[5][0]               # heading angle
            beta_crab = ssa(math.atan2(v, u))  # crab angle
            chi = psi + beta_crab       # course angle
            
            Vx = U*np.cos(chi)  # ownship velocity in x-direction
            Vy = U*np.sin(chi)  # ownship velocity in y-direction

        # Docking
            total_distance = math.sqrt((x[3][0] - wpt_pos_x[-1])**2 + (x[4][0] - wpt_pos_y[-1])**2)
            # Goal reached check
            if total_distance <= R_switch:
                alpha = 1e-10
                u_d = sat(u_d - alpha*total_distance, 5)     # u_d_min
                print('Docking')
                if total_distance <= 0.3*R_switch:
                    #ownship = shipModel(x[3], x[4], chi, U)
                    print('Stop')
                    break
            else:
                u_d = U_ref
                
            # Guidance
            chi_ref, y_e, waypoint_index = LOSchi(x[3][0], x[4][0], Delta, R_switch, wpt_pos_x, wpt_pos_y, waypoint_index)
        
            # Low pass filter
            Ad = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [-wn_d**3, -(2*zeta_d+1)*wn_d**2, -(2*zeta_d+1)*wn_d]])
            Bd = np.array([[0], [0], [wn_d**3]])
            
            xd_dot = np.dot(Ad, xd) + np.dot(Bd, chi_ref)
            chi_d = xd[0][0]
            omega_d = xd[1][0]

            # PID control: Course
            chi_error = ssa(chi_d - chi)
            chi_error_dot = (chi_error - chi_error_past) / h
            chi_error_past = chi_error
            chi_error_sum += chi_error
            
            # PID control: Surge
            surge_error = u_d - U
            surge_error_dot = (surge_error - surge_error_past) / h
            surge_error_past = surge_error
            surge_error_sum += surge_error * h
            tau_X = Kp_surge*surge_error + Kd_surge*surge_error_dot + Ki_surge*surge_error_sum
            tau_N = Kp*chi_error + Kd*chi_error_dot + Ki*chi_error_sum
            tau = np.array([[tau_X], [tau_N]])
            control = np.dot(Binv, tau)
        
            Tp = control[0]
            Ts = control[1]
            # T_max = np.ones((len(t),1)) * T_max
            # T_min = np.ones((len(t),1)) * T_min
            
            for index in range(2):
                if control[index] > T_max:              # saturation, physical limits
                    control[index] = T_max
                elif control[index] < T_min:
                    control[index] = T_min
            
            # store simulation data in a table
            # t (1), x'(2:7), chi_ref(8), chi_d(9), y_e(10), U_ref(11), u_d(12), control'(13:14)
            output = np.concatenate(([t], x.T.flatten(), [chi_ref], [chi_d], [y_e], [U_ref], [u_d], control.T.flatten()))
            if i == 1:
                output_log = output
            else:
                output_log = np.vstack((output_log, output))
            # Euler integration (k+1)
            x = x + h * WAMV_CNU(x, control)
            xd = xd + h*xd_dot
            
            if i % 10 == 0:
                path_x.append(x[3])
                path_y.append(x[4])
                
                sub1.cla()
                sub1.set(xlim=[0.0, 600.0],
                         ylim=[0.0, 300.0],
                         title='Guidance + PID control map',
                         xlabel='$x$-position [m]',
                         ylabel='$y$-position [m]')
                sub1.plot(start_X, start_Y, 'ro')
                sub1.plot(goal_X, goal_Y, 'ro')
                sub1.text(start_X + 2, start_Y,'Start', fontsize=10)
                sub1.text(goal_X + 2 ,goal_Y, 'Goal', fontsize=10)
                sub1.plot(wpt_pos_x, wpt_pos_y, 'ro', linewidth=2)
                sub1.add_patch(rect1)
                sub1.add_patch(rect2)
                
                ownship = shipModel(x[3][0], x[4][0], chi, U)
                sub1.plot(path_x, path_y, 'b.')
                sub1.text(x[3][0] + 10, x[4][0] - 10, 'USV', color='b')
                s = 'Time = {:8.2f}'.format(t)
                sub1.text(10, 270, s, fontsize=9)
                        
                sub3.set(title = 'Thrust (N)',
                         xlabel = 'time (s)',
                         ylim=[-400, 400])
                sub3.legend(['T_max', 'T_min', 'T_p', 'T_s'])
                sub3.plot(t, Tp, 'g.')
                sub3.plot(t, Ts, 'k.')
                plt.draw()
                plt.pause(0.001)
            
cid1 = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
cid2 = plt.gcf().canvas.mpl_connect('key_press_event', onkeypress)

plt.show()
