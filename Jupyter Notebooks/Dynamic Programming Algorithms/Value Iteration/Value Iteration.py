##################### Value Iteration #########################
#%% Problem 1: Sample Collection
### Time steps of decision
T = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0]

### Samples
Samples = [0, 250, 500, 750, 1000, 1250, "Succes"]

### Space of States
S_t = []
for S in [0, 250, 500, 750]:
    S_t.append((1000,S))
for t in [900, 800, 700, 600, 500, 400, 300, 200, 100, 0]:
    for S in Samples:
        S_t.append((t,S))

### Actions 
# CM for Calculated Maneuver
# IM for improvised Maneuver 
A = ["CM", "IM"]

### Transition probabilities
## Transition Probabilites will be defined as dictionaries
# Transition Probabilities for Calculated Maneuvers
p_CM = {}
for s_t in S_t:
    for s_tplus1 in S_t:
        if s_tplus1[0] == s_t[0] - 100:
            if s_t[1] != "Succes" and s_tplus1[1] != "Succes":
                if s_tplus1[1] == s_t[1] + 250 and s_t[1] < 1250:
                    p_CM[s_t, s_tplus1] = 0.3
                elif s_tplus1[1] == s_t[1] + 500 and s_t[1] < 1000:
                    p_CM[s_t, s_tplus1] = 0.2
                elif s_tplus1[1] == s_t[1] + 750 and s_t[1] < 750:
                    p_CM[s_t, s_tplus1] = 0.1
                elif s_tplus1[1] == s_t[1] and s_t[1] != "Succes":
                    p_CM[s_t, s_tplus1] = 0.4
                else: 
                    p_CM[s_t, s_tplus1] = 0
            elif s_tplus1[1] == "Succes" and s_t[1] == 1000:
                p_CM[s_t, s_tplus1] = 0.3
            elif s_tplus1[1] == "Succes" and s_t[1] == 1250:
                p_CM[s_t, s_tplus1] = 0.6
            elif s_tplus1[1] == "Succes" and s_t[1] == 750:
                p_CM[s_t, s_tplus1] = 0.1
            elif s_tplus1[1] == s_t[1] and s_t[1] == "Succes":
                p_CM[s_t, s_tplus1] = 1
            else: 
                p_CM[s_t, s_tplus1] = 0
        else:
            p_CM[s_t, s_tplus1] = 0
    
# Transition Probabilites for Improvised Maneuvers
p_IM = {}
for s_t in S_t:
    for s_tplus1 in S_t:
        if s_tplus1[0] == s_t[0] - 100:
            if s_t[1] != "Succes" and s_tplus1[1] != "Succes":
                if s_tplus1[1] == s_t[1] + 500 and s_t[1] < 1000:
                    p_IM[s_t, s_tplus1] = 0.5
                elif s_tplus1[1] == s_t[1] and s_t[1] <= 1250:
                    p_IM[s_t, s_tplus1] = 0.5
                else: 
                    p_IM[s_t, s_tplus1] = 0
            elif s_tplus1[1] == "Succes" and (s_t[1] == 1000 or s_t[1] == 1250):
                p_IM[s_t, s_tplus1] = 0.5
            elif s_tplus1[1] == s_t[1] and s_t[1] == "Succes":
                p_IM[s_t, s_tplus1] = 0.1
            else: 
                p_IM[s_t, s_tplus1] = 0
        else:
            p_IM[s_t, s_tplus1] = 0

# Consolidation of probabilities
pTrans = {"CM":p_CM, "IM": p_IM}

### Rewards
# Reward r(s_t)
r = {}
for s_t in S_t:
    r[s_t] = 0
r[(0,"Succes")] = 1

### Arbitrary Policy π
pi = {}
# The initialization will be taking calculated maneuvers in all states
for s_t in S_t:
    if s_t[0] != 0:
        pi[s_t] = "CM"

### Policy's Value
# Defining dictionary V for the value of the policy for each state in 0.5 (arbitrarily)
V = {}
for s_t in S_t:
    V[s_t] = 0.5
for samples in [0, 250, 500, 750, 1000, 1250, "Succes"]:
    V[(0,samples)] = 0
    
### Value Iteration
# Threshold theta indicates the desired accuracy of the finded value throught iterations
theta = 0.005

# Variable tracking the number of iterations
num_iter = 0

# Initializing delta in 10 so the loop can start
delta = 10

# Iterations
while delta > theta:
    delta = 0
    for s_t in S_t:
        if s_t[0] != 0:
            v = V[s_t]
            Vmax = 0
            argmax = ""
            for a in A:
                value = 0
                for s_tplus1 in S_t:
                    value += pTrans[a][s_t, s_tplus1] * (r[s_tplus1] + V[s_tplus1]) 
                if value > Vmax:
                    Vmax = value
                    argmax = a
            V[s_t] = Vmax
            pi[s_t] = argmax
            delta = max(delta, abs(v - V[s_t]))

for t in T:
    if t != 0:
        if pi[(t,0)] == "":
            pi[(t,0)] = "Mission Failed"
        print(f"The optimal decision for {t} meters when having 0 samples is: {pi[(t,0)]}")

print("\n")
print(f"The probability of succes at the beggining of the mission is:{round(V[(1000,0)],3)}")

#%% Problem 2: Aqueduct
### States
S = list(range(3,11))

### Decisions
A = ["Send", "Don't Send"]

### Transition probabilities
# Send the cleaning team
p_SEND = {}
for s_t in S:
    for s_tplus1 in S:
        if s_tplus1 == 3:
            p_SEND[s_t, s_tplus1] = 1
        else:
            p_SEND[s_t, s_tplus1] = 0

# Dont's send the cleaning team 
p_DONTS = {}
for s_t in S:
    for s_tplus1 in S:
        if s_tplus1 >= s_t:
            p_DONTS[s_t, s_tplus1] = 1/(11-s_t)
        else:
            p_DONTS[s_t, s_tplus1] = 0

# Consolidation of probabilities
pTrans = {"Send": p_SEND, "Don't Send": p_DONTS}

### Costs
c = {}
for s_t in S:
    c[s_t, "Send"] = 12 + s_t * 0.01 * 200
    c[s_t, "Don't Send"] =  s_t * 0.01 * 200

### Discount Factor 
gamma = 0.95

### Arbitrary policy π
pi = {}
# The initialization will be sending the team in all states
for s_t in S:
    pi[s_t] = "Send"


### Policy's Value
V = {}
# Initializing dictionary V for the value of the policy for each state in 300 (arbitrarily)
for s_t in S:
    V[s_t] = 300
 
### Value Iteration
# Threshold theta indicates the desired accuracy of the finded value throught iterations
theta = 0.005

# Variable tracking the number of iterations
num_iter = 0

# Iterations
# Initializing delta in 10 so the loop can start
delta = 10
# Iterations
while delta > theta:
    delta = 0
    for s_t in S:
            v = V[s_t]
            Vmin = 500
            argmin = ""
            for a in A:
                value = 0
                for s_tplus1 in S:
                    value += pTrans[a][s_t, s_tplus1] * (c[s_t,a] + gamma * V[s_tplus1]) 
                if value < Vmin:
                    Vmin = value
                    argmin = a
            V[s_t] = Vmin
            pi[s_t] = argmin
            delta = max(delta, abs(v - V[s_t]))

    ### Update the number of iterations
    num_iter += 1
    
print(f"Number of iterations before achieving optimality: {num_iter}") 

print("The optimal decision to make is:")
for s in S:
    print(f"- If there are {s} liters of trash, {pi[s]} the Team")
    
print("\n")
print("The expected cost for every state is:")
for s in S:
    print(f"- When there are {s} liters of trash: {round(V[s],3)} Million Pesos")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

