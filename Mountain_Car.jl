# Mountain_Car.jl
#import Pkg

using RxInfer, Plots
import HypergeometricFunctions: _₂F₁
import RxInfer.ReactiveMP: getrecent, messageout



# Create the physics of the Mountain Car problem

function create_physics(; engine_force_limit = 0.04, friction_coefficient = 0.1)
    # Engine force as function of action
    Fa = (a::Real) -> engine_force_limit * tanh(a) 

    # Friction force as function of velocity
    Ff = (y_dot::Real) -> -friction_coefficient * y_dot 
    
    # Gravitational force (horizontal component) as function of position
    Fg = (y::Real) -> begin
        if y < 0
            0.05*(-2*y - 1)
        else
            0.05*(-(1 + 5*y^2)^(-0.5) - (y^2)*(1 + 5*y^2)^(-3/2) - (y^4)/16)
        end
    end
    
    # The height of the landscape as a function of the horizontal coordinate
    height = (x::Float64) -> begin
        if x < 0
            h = x^2 + x
        else
            h = x * _₂F₁(0.5,0.5,1.5, -5*x^2) + x^3 * _₂F₁(1.5, 1.5, 2.5, -5*x^2) / 3 + x^5 / 80
        end
        return 0.05*h
    end

    return (Fa, Ff, Fg,height)
end;

# Create the world of the Mountain Car problem
# The world is defined by the forces acting on the car and the initial state
# The initial state is defined by the initial position and velocity of the car
# The world is a tuple of two functions: execute and observe
# The execute function takes an action and computes the next state of the car
# The observe function returns the current state of the car (position and velocity)
# The state is reset after each step to allow for multiple steps in a row
# The initial position is set to -0.5 and the initial velocity is set to 0.0 by default


function create_world(; Fg, Ff, Fa, initial_position = -0.5, initial_velocity = 0.0)

    y_t_min = initial_position
    y_dot_t_min = initial_velocity
    
    y_t = y_t_min
    y_dot_t = y_dot_t_min
    
    execute = (a_t::Float64) -> begin
        # Compute next state
        y_dot_t = y_dot_t_min + Fg(y_t_min) + Ff(y_dot_t_min) + Fa(a_t)
        y_t = y_t_min + y_dot_t
    
        # Reset state for next step
        y_t_min = y_t
        y_dot_t_min = y_dot_t
    end
    
    observe = () -> begin 
        return [y_t, y_dot_t]
    end
        
    return (execute, observe)
end


# Create the world of the Mountain Car problem

engine_force_limit   = 0.04
friction_coefficient = 0.1

Fa, Ff, Fg, height = create_physics(
    engine_force_limit = engine_force_limit,
    friction_coefficient = friction_coefficient
);
initial_position = -0.5
initial_velocity = 0.0

x_target = [0.5, 0.0] 

valley_x = range(-2, 2, length=400)
valley_y = [ height(xs) for xs in valley_x ]
plot(valley_x, valley_y, title = "Mountain valley", label = "Landscape", color = "black")
scatter!([ initial_position ], [ height(initial_position) ], label="initial car position")   
scatter!([x_target[1]], [height(x_target[1])], label="camping site")


# Naive policy: always apply full power to the right
N_naive  = 100 # Total simulation time
pi_naive = 100.0 * ones(N_naive) # Naive policy for right full-power only

# Let there be a world
(execute_naive, observe_naive) = create_world(; 
    Fg = Fg, Ff = Ff, Fa = Fa, 
    initial_position = initial_position, 
    initial_velocity = initial_velocity
);

y_naive = Vector{Vector{Float64}}(undef, N_naive)
for t = 1:N_naive
    execute_naive(pi_naive[t]) # Execute environmental process
    y_naive[t] = observe_naive() # Observe external states
end

animation_naive = @animate for i in 1:N_naive
    plot(valley_x, valley_y, title = "Naive policy", label = "Landscape", color = "black", size = (800, 400))
    scatter!([y_naive[i][1]], [height(y_naive[i][1])], label="car")
    scatter!([x_target[1]], [height(x_target[1])], label="goal")   
end

# The animation is saved and displayed as markdown picture for the automatic HTML generation
gif(animation_naive, "ai-mountain-car-naive.gif", fps = 24, show_msg = false);


### Active inference policy

#Defining Handcrafted Model
@model function mountain_car(m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min, T, Fg, Fa, Ff, engine_force_limit)

    # ================================
    # 1. Physics Functions (Fixed)
    # ================================

    # Environment dynamics (without engine control): gravity + friction
    g = (s_t_min::AbstractVector) -> begin 
        s_t = similar(s_t_min)            # s_t = [position, velocity]
        s_t[2] = s_t_min[2] + Fg(s_t_min[1]) + Ff(s_t_min[2])  # update velocity
        s_t[1] = s_t_min[1] + s_t[2]      # update position
        return s_t
    end

    # Engine control model: maps action u to velocity change
    h = (u::AbstractVector) -> [0.0, Fa(u[1])]

    # Inverse engine control model: maps velocity change to action u
    h_inv = (delta_s_dot::AbstractVector) -> [
        atanh(clamp(delta_s_dot[2], -engine_force_limit + 1e-3, engine_force_limit - 1e-3) / engine_force_limit)
    ]

    # ================================
    # 2. Hyperparameters
    # ================================

    Gamma = 1e4 * diageye(2)   # Precision of state transition (1/variance)
    Theta = 1e-4 * diageye(2)  # Variance of sensory observation

    # ================================
    # 3. Prior over Initial State
    # ================================

    s_t_min ~ MvNormal(mean = m_s_t_min, cov = V_s_t_min)  # prior belief over initial position and velocity

    s_k_min = s_t_min  # Initialize state for k=1

    local s  # Store future states

    # ================================
    # 4. Inference Loop over Future Horizon T
    # ================================

    for k in 1:T

        # ----------------------------
        # Prior over Control (Action)
        # ----------------------------
        u[k] ~ MvNormal(mean = m_u[k], cov = V_u[k])  # agent's belief over action at time k

        # ----------------------------
        # Apply Engine Model (forward and inverse)
        # ----------------------------
        u_h_k[k] ~ h(u[k]) where {
            meta = DeltaMeta(method = Linearization(), inverse = h_inv)
        }

        # ----------------------------
        # Apply Physical Transition Model
        # ----------------------------
        s_g_k[k] ~ g(s_k_min) where {
            meta = DeltaMeta(method = Linearization())
        }

        # ----------------------------
        # Combine physics + control
        # ----------------------------
        u_s_sum[k] ~ s_g_k[k] + u_h_k[k]

        # ----------------------------
        # Predict Next State (Posterior)
        # ----------------------------
        s[k] ~ MvNormal(mean = u_s_sum[k], precision = Gamma)

        # ----------------------------
        # Likelihood: what the agent expects to observe given its internal state
        # ----------------------------
        x[k] ~ MvNormal(mean = s[k], cov = Theta)

        # ----------------------------
        # Prior over Preferred Observation (Goal)
        # Pulls x[k] toward goal position and velocity
        # ----------------------------
        x[k] ~ MvNormal(mean = m_x[k], cov = V_x[k])

        # Move to next timestep
        s_k_min = s[k]
    end

    # Return final belief over state trajectory
    return (s, )
end



# Create the agent that will use the model to perform active inference
# The agent will have a compute function that takes an action and an observation
# The agent will have an act function that returns the most probable action
# The agent will have a slide function that moves the belief one step forward in time
# The agent will have a future function that predicts the future states
# The agent will have a create_agent function that takes the parameters of the model and returns the agent
function create_agent(; T = 20, Fg, Fa, Ff, engine_force_limit, x_target, initial_position, initial_velocity)

    # Constants for numerical stability and variance scaling
    huge = 1e6     # Very large variance (uninformative prior)
    tiny = 1e-6    # Very small variance (strong confidence)

    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # CONTROL PRIORS (action beliefs for T steps into the future)
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    Epsilon = fill(huge, 1, 1)  # Very uncertain prior
    m_u = [ [0.0] for k = 1:T ]  # Initial mean actions: all zeros
    V_u = [ Epsilon for k = 1:T ]  # Initial action variances: uninformative

    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # GOAL PRIOR (agent believes it should reach the goal at final timestep)
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    Sigma = 1e-4 * diageye(2)     # Goal belief uncertainty: confident
    m_x = [ zeros(2) for k = 1:T ]  # Predicted observations (position, velocity)
    V_x = [ huge * diageye(2) for k = 1:T ]  # Initially uncertain
    V_x[end] = Sigma  # Only final step is constrained to goal

    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # BRAIN STATE PRIOR (initial belief over position and velocity)
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    m_s_t_min = [initial_position, initial_velocity]
    V_s_t_min = tiny * diageye(2)

    # Store last inference result
    result = nothing

    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # COMPUTE (Act-Execute-Observe + Inference)
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    compute = (upsilon_t::Float64, y_hat_t::Vector{Float64}) -> begin
        # Clamp the control (action taken in real world)
        m_u[1] = [upsilon_t]
        V_u[1] = fill(tiny, 1, 1)

        # Clamp the observation (what the agent sees)
        m_x[1] = y_hat_t
        V_x[1] = tiny * diageye(2)

        # Bundle all inputs to pass to the generative model
        data = Dict(
            :m_u       => m_u,
            :V_u       => V_u,
            :m_x       => m_x,
            :V_x       => V_x,
            :m_s_t_min => m_s_t_min,
            :V_s_t_min => V_s_t_min
        )

        # Build the model and run inference
        model = mountain_car(
            T = T,
            Fg = Fg,
            Fa = Fa,
            Ff = Ff,
            engine_force_limit = engine_force_limit
        )

        # Run variational inference (posterior beliefs)
        result = infer(model = model, data = data)
    end

    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # ACT (select the most probable action from posterior)
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    act = () -> begin
        if result !== nothing
            return mode(result.posteriors[:u][2])[1]  # Select best next action
        else
            return 0.0  # Default action if inference hasn’t run yet
        end
    end

    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # FUTURE (predict future states)
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    future = () -> begin
        if result !== nothing
            return getindex.(mode.(result.posteriors[:s]), 1)  # Get position from all future states
        else
            return zeros(T)
        end
    end

    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # SLIDE (move belief one step forward in time)
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    slide = () -> begin
        model  = RxInfer.getmodel(result.model)
        (s, )  = RxInfer.getreturnval(model)
        varref = RxInfer.getvarref(model, s)
        var    = RxInfer.getvariable(varref)

        # Retrieve posterior state belief at second time step
        slide_msg_idx = 3  # Model-dependent index
        (m_s_t_min, V_s_t_min) = mean_cov(getrecent(messageout(var[2], slide_msg_idx)))

        # Shift beliefs forward (simulate the next time step)
        m_u = circshift(m_u, -1);     m_u[end] = [0.0]
        V_u = circshift(V_u, -1);     V_u[end] = Epsilon
        m_x = circshift(m_x, -1);     m_x[end] = x_target
        V_x = circshift(V_x, -1);     V_x[end] = Sigma
    end

    # Return the functional interface for using the agent
    return (compute, act, slide, future)
end




# 🌍 STEP 1: Create the environment
# The environment simulates the external world – i.e., the mountain valley with gravity, friction, and engine force.
(execute_ai, observe_ai) = create_world(
    Fg = Fg,           # Gravitational force function
    Ff = Ff,           # Friction force function
    Fa = Fa,           # Engine force function
    initial_position = initial_position,
    initial_velocity = initial_velocity
)

# ⏱️ Define the agent's prediction horizon (how far ahead it predicts)
T_ai = 50  # This is the agent’s "mental timeline" for planning

# 🧠 STEP 2: Create the agent
# The agent has internal beliefs over states and goals, and performs inference to act adaptively.
(compute_ai, act_ai, slide_ai, future_ai) = create_agent(
    T  = T_ai,                        # Number of future steps it plans over
    Fa = Fa, Fg = Fg, Ff = Ff,        # Physics model functions
    engine_force_limit = engine_force_limit,
    x_target = x_target,             # Target position (goal)
    initial_position = initial_position,
    initial_velocity = initial_velocity
)

# 🔁 STEP 3: Run the simulation
N_ai = 100  # Number of time steps (episodes) to simulate

# Buffers to store the simulation data
agent_a = Vector{Float64}(undef, N_ai)         # Actions taken by the agent
agent_f = Vector{Vector{Float64}}(undef, N_ai) # Predicted future positions
agent_x = Vector{Vector{Float64}}(undef, N_ai) # Observed positions (true environment)

# 🔄 Active Inference loop
for t = 1:N_ai
    agent_a[t] = act_ai()               # 🔧 Select action based on internal beliefs (action = mode of inferred control)
    agent_f[t] = future_ai()            # 🔮 Predict future positions from the model
    execute_ai(agent_a[t])              # 🚗 Act on the environment (update hidden physical state)
    agent_x[t] = observe_ai()           # 👀 Observe new state (position + velocity)
    compute_ai(agent_a[t], agent_x[t])  # 🧠 Perform inference: update beliefs (posteriors) using new observation
    slide_ai()                          # ⏩ Slide forward the internal belief window to next timestep
end

# 🎞️ STEP 4: Create animation of the agent's behavior
animation_ai = @animate for i in 1:N_ai
    # Plot 1: Car in the valley
    pls = plot(valley_x, valley_y,
        title = "Active Inference Results",
        label = "Landscape",
        color = "black"
    )
    
    # Current car position
    scatter!(pls, [agent_x[i][1]], [height(agent_x[i][1])], label = "Car")
    
    # Goal position
    scatter!(pls, [x_target[1]], [height(x_target[1])], label = "Goal")
    
    # Predicted future trajectory
    scatter!(pls, agent_f[i], height.(agent_f[i]),
        label = "Predicted Future",
        alpha = map(j -> 0.5 / j, 1:T_ai)  # Older predictions are more transparent
    )

    # Plot 2: Engine force over time
    pef = plot(
        Fa.(agent_a[1:i]), 
        title = "Engine Force (Agent Actions)",
        xlim = (0, N_ai),
        ylim = (-0.05, 0.05),
        label = "Force"
    )
    
    # Combine plots: landscape on top, engine force below
    plot(pls, pef, size = (800, 400))
end

# 💾 STEP 5: Export the animation
gif(animation_ai, "ai-mountain-car-ai.gif", fps = 24, show_msg = false)