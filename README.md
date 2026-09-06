---
title: "Neural-Guided Monte Carlo Tree Search for Economic Equipment Placement in Active Distribution Networks: An IEEE 33-Bus Case Study"
author: "Bita Nayeri"
---

Department of Electrical Engineering (Power and Control), Shahid Beheshti University

## Abstract

Selecting equipment types and locations in an active distribution network requires discrete investment decisions to be assessed against time-dependent operating costs and network constraints. This paper presents an implementation of reinforcement learning with neural-guided Monte Carlo tree search (MCTS) for fixed-capacity equipment placement in the radial IEEE 33-bus system. Candidate technologies comprise energy storage, gas-fired generation, static var compensators, and capacitor banks. A policy-value network guides tree simulations, while selected terminal placements are evaluated by a multiperiod DistFlow-based operating model solved with IPOPT. Annual economic cost is reported separately from artificial constraint penalties and current regularization. Final designs undergo a further solve with artificial slacks fixed to zero and a separate numerical constraint audit. In one 400-episode training run, 341 episodes produced usable numerical outcomes. The best recorded training design, containing six 0.5-MW gas units, reduced annual economic cost from USD 571,353.16 to USD 403,888.92 after final verification, corresponding to a 29.31% saving. The direct final-checkpoint trajectory achieved a separate saving of 6.61%. A single-trial benchmark under a common cap of eight new operating-model evaluations returned costs of USD 491,841.63, USD 522,415.94, and USD 489,176.85 for random, budgeted greedy, and neural-guided search, respectively. Fixed-placement redispatch at electricity-price multipliers of 0.8, 1.0, and 1.2 yielded savings of 18.73%, 29.31%, and 37.85% for the best training design. These findings establish case-specific computational feasibility and economic improvement; they do not establish global optimality or statistical superiority.

**Keywords:** Active distribution network; reinforcement learning; Monte Carlo tree search; equipment placement; optimal power flow; electricity-price sensitivity.

# 1. Introduction

Distributed generation, storage, and controllable reactive-power resources connect investment decisions directly to the operating conditions of distribution feeders. A resource installed at one bus can alter upstream power exchange, branch losses, voltage limits, and the amount of renewable energy that can be accommodated. Its economic value consequently depends on both location and dispatch. Comparing investment costs alone can overlook operating benefits, while assessing only an operating snapshot can overlook seasonal and hourly constraints.

The resulting planning task combines a discrete choice of technology and bus with a continuous network operating problem. The branch-flow formulation introduced for radial distribution analysis provides a useful connection between power, voltage, and losses [1]. Its nonlinear current-power relation nevertheless complicates repeated evaluation. Convex relaxation can simplify this relation, but exactness depends on model conditions and cannot be presumed for every extension of an optimal power flow (OPF) problem [2]. A planning algorithm therefore needs both an effective way to propose designs and a credible way to evaluate the designs it reports.

Monte Carlo tree search allocates simulations adaptively across a sequential decision tree [3]. Neural policy and value approximations can guide exploration and replace expensive evaluations within that tree, as illustrated by the broader policy-value search framework used in AlphaZero [4]. In active distribution network planning, Zhang et al. [5] already combined MCTS, reinforcement learning, and a three-output neural network to pursue operating-performance targets. Their study used the IEEE 33-bus system and considered storage, gas generation, static var compensation, and capacitor installation. The present work is an implementation and adaptation of that research direction, rather than a claim to introduce MCTS-based distribution planning.

The practical focus here is the distinction between the cost of a candidate, the numerical objective used to obtain its dispatch, and the evidence supporting its feasibility. This distinction matters when artificial slacks are used during search or when a relaxed network relation is encouraged toward equality through regularization. It also matters when the most economical design encountered during training differs from the output of the final learned policy. Reporting only the better of those outcomes can obscure what the trained model actually produces.

Electricity prices introduce a further distinction. A time-dependent tariff may be an externally specified input to a planning model or the outcome of a market-clearing mechanism. Li and Gao [6], for example, study real-time pricing through a convex-hull market formulation. This paper instead prescribes purchase and sale prices. It evaluates how fixed designs operate under those prices; it does not optimize the tariff or model price-responsive demand.

The study makes three contributions at the implementation and evaluation level. First, it couples neural-guided placement search to a seasonal operating model with explicit annual cost accounting and a separate final constraint audit. Second, it reports the direct checkpoint result, the best design recorded during training, and a limited-budget benchmark as distinct experiments. Third, it evaluates the sensitivity of a fixed candidate set to simultaneous changes in import and export prices. The numerical evidence is deliberately restricted to one trained checkpoint and one feeder. The research question is whether this implementation can find less costly designs that satisfy its stated numerical acceptance criteria, and what qualifications are needed when interpreting those designs.

# 2. Planning and operating model

## 2.1. Decision variables, feeder, and representative days

Let $\mathcal{N}=\{1,\ldots,33\}$ denote the buses, with bus 1 representing the substation, and let $\mathcal{E}$ contain the 32 directed feeder branches. A binary placement indicator $x_i^h$ specifies whether one device of technology $h$ is installed at candidate bus $i\in\{2,\ldots,33\}$. Technologies are $h\in\{\mathrm{ess},\mathrm{gas},\mathrm{svc},\mathrm{cb}\}$. A technology cannot be installed twice at the same bus, although different technologies may share a bus. Device capacities are fixed. The placement indicators are set by the search algorithm before each operating-model solve and are not integer decision variables within IPOPT.

The electrical bases are 1 MVA and 12.66 kV. Existing 0.5-MW photovoltaic (PV) units are located at buses 9, 11, 14, 17, 19, 21, 23, and 29, following the study configuration in [5]. Their investment cost is treated as sunk and excluded from incremental planning cost. Inverter reactive-power support remains available in every experiment, including the base case. The base case therefore means no newly installed planning equipment; it already contains PV and controllable inverter operation.

Operation is represented by four independent 24-hour seasonal days. Their annual weights are 92, 93, 90, and 90 days for spring, summer, autumn, and winter, respectively. Hour index $t\in\{0,\ldots,95\}$ identifies a representative operating point, and $w_t$ is its seasonal day weight. The time step is $\Delta t=1$ h. Load and PV profiles are prescribed study inputs, not a statistically fitted uncertainty model. Their numerical arrays, together with feeder data, are supplied in the accompanying source package.

## 2.2. Economic accounting

For interest rate $r=0.05$ and lifetime $n=20$ years, the capital recovery factor and annualized incremental investment are

$$
\operatorname{CRF}=\frac{r(1+r)^n}{(1+r)^n-1},\qquad
C_{\mathrm{inv}}=\operatorname{CRF}\sum_{i,h}c_h^{\mathrm{inv}}K_hx_i^h.\qquad(1)
$$

Here, $K_h$ is the fixed rating expressed in the unit used by its investment coefficient: MW for storage and gas generation, and MVAr for SVCs and capacitors. Storage investment is parameterized per MW in this implementation, with a fixed associated energy rating. It is not a separate optimization of power and energy capacity.

All power variables in the operating equations are per unit. Multiplication by $S_B$, the 1-MVA base, converts active power to MW for the cost and energy calculations. The annual economic cost is

$$
\begin{aligned}
C_{\mathrm{econ}}&=C_{\mathrm{inv}}+S_B\Delta t\sum_t w_t
\left(c_t^{\mathrm{grid}}+\sum_i c_{i,t}^{\mathrm{loc}}\right),\\
c_t^{\mathrm{grid}}&=\lambda_t^{\mathrm{buy}}p_t^{\mathrm{imp}}-
\lambda_t^{\mathrm{sell}}p_t^{\mathrm{exp}},\\
c_{i,t}^{\mathrm{loc}}&=c_{\mathrm{gas}}p_{i,t}^{\mathrm{gas}}
+c_{\mathrm{deg}}(p_{i,t}^{\mathrm{ch}}+p_{i,t}^{\mathrm{dis}})
+c_{\mathrm{RES}}p_{i,t}^{\mathrm{curt}}
+c_{\mathrm{AUL}}p_{i,t}^{\mathrm{shed}}.
\end{aligned}\qquad(2)
$$

The gas coefficient includes USD 50/MWh for fuel and USD 20/MWh for the modeled emissions charge. Storage throughput costs USD 20/MWh; curtailed PV and unserved load carry economic coefficients of USD 500/MWh and USD 5,000/MWh, respectively. These last two terms remain part of economic cost even when a design passes its performance thresholds. They differ from penalties on artificial constraint slacks. Losses affect power balance and grid purchases and are not charged a second time as an independent energy expense.

The numerical operating objective is

$$
J=C_{\mathrm{econ}}+C_{\mathrm{pen}}+
\alpha\sum_{(i,j)\in\mathcal E,t}w_t\ell_{ij,t},\qquad \alpha=0.1.\qquad(3)
$$

Nonnegative slacks soften voltage, branch-current, substation, and end-of-day storage constraints during candidate evaluation. Each penalty category uses a coefficient of $10^7$. The last term encourages smaller squared currents and tighter branch-flow relations. Candidate costs are computed from the dispatch returned by this regularized local solve. Consequently, reporting $C_{\mathrm{econ}}$ separately does not mean that its dispatch has been independently optimized for the unregularized economic objective.

**Table 1. Equipment ratings and investment assumptions.**

| Technology | Fixed rating per installation | Investment coefficient | Operating capability |
|---|---|---|---|
| Energy storage | 0.10 MW / 0.50 MWh | 200,000 USD/MW | 0.10-MWh minimum energy; 0.90 charge and discharge efficiencies |
| Gas generator | 0.50 MW | 30,000 USD/MW | 0.625-MVA apparent-power capability |
| SVC | 0.50 MVAr | 500,000 USD/MVAr | Continuous absorption or injection |
| Capacitor bank | 0.50 MVAr | 40,000 USD/MVAr | Continuous fraction of voltage-dependent reactive output |

The investment coefficients and principal device sizes follow the numerical assumptions reported in [5]. Converter margins, price profiles, annualization, and the explicit operating charges are settings of the present implementation. These coefficients are study inputs rather than verified contemporary procurement prices.

## 2.3. Network and device constraints

For branch $(i,j)$, $P_{ij,t}$ and $Q_{ij,t}$ denote sending-end powers, $\ell_{ij,t}=|I_{ij,t}|^2$, and $v_{i,t}=|V_{i,t}|^2$. Define $\delta_{i1}=1$ only at the substation. The nodal balances explicitly include the substation injection:

$$
\begin{aligned}
\sum_{k:(k,i)\in\mathcal E}(P_{ki,t}-r_{ki}\ell_{ki,t})+
\delta_{i1}p_t^{\mathrm{sub}}+p_{i,t}^{\mathrm{loc}}
&=\sum_{j:(i,j)\in\mathcal E}P_{ij,t}+p_{i,t}^{\mathrm{serv}},\\
\sum_{k:(k,i)\in\mathcal E}(Q_{ki,t}-x_{ki}\ell_{ki,t})+
\delta_{i1}q_t^{\mathrm{sub}}+q_{i,t}^{\mathrm{loc}}
&=\sum_{j:(i,j)\in\mathcal E}Q_{ij,t}+q_{i,t}^{\mathrm{serv}}.
\end{aligned}\qquad(4)
$$

Local active injection comprises available PV minus curtailment, gas output, and net storage discharge. Reactive injection includes gas, PV and storage inverters, SVCs, and capacitor banks. Served active demand equals forecast demand minus load shedding. Reactive demand is reduced by the same fraction as active demand, preserving the specified load power factor. Curtailment and shedding cannot exceed the corresponding available PV and active demand.

The DistFlow voltage relation and relaxed current-power constraint are

$$
\begin{aligned}
v_{j,t}&=v_{i,t}-2(r_{ij}P_{ij,t}+x_{ij}Q_{ij,t})
+(r_{ij}^2+x_{ij}^2)\ell_{ij,t},\\
P_{ij,t}^2+Q_{ij,t}^2&\leq v_{i,t}\ell_{ij,t}.
\end{aligned}\qquad(5)
$$

Hard limits are $0.95^2\leq v_{i,t}\leq1.05^2$, $v_{1,t}=1$, and substation apparent power no greater than 10 MVA. Every branch uses a 5-MVA study rating converted to a current limit at the nominal voltage; this is an explicit assumption because conductor ampacities are not supplied by the standard feeder data used here. It is not evidence that the corresponding physical conductors have that rating.

Import and export are nonnegative and satisfy

$$
p_t^{\mathrm{sub}}=p_t^{\mathrm{imp}}-p_t^{\mathrm{exp}},\qquad
p_t^{\mathrm{imp}}p_t^{\mathrm{exp}}\leq\epsilon_{\mathrm{comp}},
\qquad\epsilon_{\mathrm{comp}}=10^{-7}.\qquad(6)
$$

Exports are additionally limited by actual PV production plus gas output in that hour. This is a modeling restriction, including when storage is available. Hourly purchase prices are a base 24-hour profile between USD 20/MWh and USD 120/MWh, multiplied by seasonal factors 1.0, 1.5, 1.0, and 1.2. Thus the maximum purchase price in the main study is USD 180/MWh. The sale tariff is USD 19.99/MWh throughout. Demand remains exogenous.

For an installed storage unit, both charging and discharging power are bounded separately by its 0.10-MW rating, and their product is constrained by the same complementarity tolerance. With energy expressed in MWh,

$$
e_{i,t}=e_{i,t-1}+S_B\Delta t
\left(\eta_{\mathrm{ch}}p_{i,t}^{\mathrm{ch}}-
\frac{p_{i,t}^{\mathrm{dis}}}{\eta_{\mathrm{dis}}}\right),
\qquad\eta_{\mathrm{ch}}=\eta_{\mathrm{dis}}=0.90.\qquad(7)
$$

The energy bounds are $0.10x_i^{\mathrm{ess}}\leq e_{i,t}\leq0.50x_i^{\mathrm{ess}}$. Each representative day starts and ends at $0.25x_i^{\mathrm{ess}}$ MWh. This prevents energy transfer between independent seasonal days. Net storage active power and reactive power share a 0.11-MVA converter capability circle.

Gas output ranges independently each hour from zero to 0.50 MW, with a 0.625-MVA capability circle for active and reactive power. No commitment, startup, minimum-up-time, or ramp constraints are imposed. PV inverters use an apparent-power rating 1.10 times the installed PV active-power rating. An installed SVC supplies or absorbs up to 0.50 MVAr. Capacitor output is represented as $q_{i,t}^{\mathrm{cb}}=(0.50/S_B)u_{i,t}^{\mathrm{cb}}v_{i,t}$, with $0\leq u_{i,t}^{\mathrm{cb}}\leq x_i^{\mathrm{cb}}$. The capacitor setting is continuous and therefore does not represent an integer number of switched steps.

Although (5) uses a conic relaxation of the current-power relation, the complementarity constraints and voltage-dependent capacitor expression retain nonconvexity in the complete operating model. IPOPT is used as a local nonlinear solver [7]. Neither an optimal solver termination label nor the presence of a conic relation provides a global planning certificate.

# 3. Neural-guided placement search and final verification

## 3.1. State, actions, and learning targets

The sequential state is a 128-component binary placement vector, ordered by candidate bus and technology. Initially, 128 installation actions and one STOP action are available. A legal installation sets one previously zero component to one. STOP leaves the placement unchanged and terminates the episode. Each trajectory is limited to eight decisions; selecting STOP can therefore produce fewer than eight installations. This limit controls computational effort and is not a physical requirement to install eight devices.

The implementation is a single-agent planning process. The term self-play refers to generating the agent's own planning trajectories, not to competition between two players. Intermediate installation rewards are zero. For a numerically solved design that passes the compliance criteria, the terminal target is

$$
z=0.5+0.5\tanh\left(
\frac{C_{\mathrm{base}}-C_{\mathrm{econ}}}{0.25C_{\mathrm{base}}}
\right).\qquad(8)
$$

This mapping assigns 0.5 to a compliant design with base-case cost and is monotone in the cost saving. It preserves the ordering of deterministic compliant designs. It is not a general proof of policy invariance for stochastic returns. The distinction is relevant because arbitrary reward transformations need not preserve an underlying decision objective [8]. Numerically successful but noncompliant outcomes receive $z=-1$. Solver failures and timeouts are excluded from replay, because a numerical failure does not by itself identify a physically inferior placement.

Two auxiliary targets describe PV curtailment and load shedding over the four representative days. Each is its unweighted representative energy divided by 0.10 MWh and clipped to $[0,1]$. Annual weights are used for costs, but not for these two acceptance targets. The same terminal value and auxiliary targets are assigned to all recorded states in the trajectory.

## 3.2. Network, tree search, and optimization

The network has two fully connected hidden layers of 128 ReLU units. Its policy head has 129 logits, its value head has a 64-unit ReLU layer followed by a scalar tanh output, and its auxiliary head has a 64-unit ReLU layer followed by two sigmoid outputs. The model contains 66,372 trainable parameters. The auxiliary output provides a training signal; acceptance of a design is determined by the operating-model audit.

Search masks illegal actions and normalizes the remaining policy probabilities. PUCT selection chooses the action maximizing

$$
Q(s,a)+c_{\mathrm{puct}}P_\theta(a\mid s)
\frac{\sqrt{N(s)}}{1+N(s,a)},\qquad c_{\mathrm{puct}}=5.\qquad(9)
$$

Here $Q$ is the mean backed-up value, $N(s,a)$ is an edge visit count, and $P_\theta$ is the legal-action prior. Values are propagated without alternating their signs, since every decision is made for the same planner. During training, root priors mix network probabilities with Dirichlet noise using concentration 0.3 and mixing weight 0.25. The STOP prior is assigned a floor of 0.05, with the remaining probabilities rescaled. A fresh search tree is built for each executed decision.

Training uses 500 tree simulations per decision. Their terminal and nonterminal leaf values are approximated by the network; IPOPT is not called inside those tree simulations. After STOP or the trajectory limit, the selected placement receives a physical-model evaluation. This design limits the number of expensive operating-model calls but makes search quality dependent on the learned value approximation. Figure 1 summarizes the separation between candidate generation, learning, and final verification.

![Figure 1. Implemented workflow. The auxiliary performance head supports training; final acceptance is based on a separate operating-model audit.](figures/fig1_workflow.png){width=6.5in}

The policy target is derived from visit counts, $\pi(a\mid s)\propto N(s,a)^{1/\tau}$. The training temperature is one throughout the executed eight-decision horizon. Replay contains up to 10,000 state records. Once at least 32 are available, four random minibatch updates are performed after each usable episode. For batch size $D$, the implemented loss is

$$
\begin{aligned}
\mathcal L&=\mathcal L_{\pi}+0.5\mathcal L_v+0.1\mathcal L_g,\\
\mathcal L_{\pi}&=-\frac{1}{D}\sum_{d=1}^{D}\sum_a\pi_d(a)\log p_{\theta,d}(a),\\
\mathcal L_v&=\frac{1}{D}\sum_{d=1}^{D}(z_d-v_{\theta,d})^2,\\
\mathcal L_g&=\frac{1}{2D}\sum_{d=1}^{D}\sum_{k=1}^{2}(g_{d,k}-\hat g_{\theta,d,k})^2.
\end{aligned}\qquad(10)
$$

The factor $2D$ in the auxiliary term follows the code's mean-squared-error reduction across both batch samples and the two outputs. Adam uses a learning rate of $10^{-4}$ and weight decay of $10^{-4}$, with gradient norm clipped to one. A fixed random seed of zero is used for the reported run.

## 3.3. Numerical acceptance and hard re-evaluation

IPOPT is configured with an iteration limit of 5,000, a CPU-time limit of 300 s per solve, a main tolerance of $10^{-7}$, and an acceptable tolerance of $10^{-6}$. Bound relaxation is disabled, original bounds are honored, and adaptive barrier updates and gradient-based scaling are used. Initialization uses previously solved base-case primal values where available. This should be distinguished from a complete primal-dual warm-start procedure.

A separate audit recomputes constraint residuals, physical-limit violations, artificial slacks, daily storage balance, complementarity products, representative curtailment and shedding, and current-power gaps. At every branch-hour point, the absolute and relative gaps are

$$
\begin{aligned}
g_{ij,t}^{\mathrm{abs}}&=
\left|\ell_{ij,t}v_{i,t}-P_{ij,t}^2-Q_{ij,t}^2\right|,\\
g_{ij,t}^{\mathrm{rel}}&=
\frac{g_{ij,t}^{\mathrm{abs}}}{\max\{|\ell_{ij,t}v_{i,t}|,\,|P_{ij,t}^2+Q_{ij,t}^2|,\,10^{-8}\}}.
\end{aligned}\qquad(11)
$$

A point passes when its absolute gap is no greater than $2.5\times10^{-5}$ per-unit squared **or** its relative gap is no greater than $10^{-4}$. The absolute threshold is the square of 0.005 per-unit apparent power, a 5-kVA scale on the selected base. It is a study tolerance, not a formal bound on the error of a recovered AC solution. Using both criteria prevents a tiny absolute discrepancy near zero flow from being rejected solely because its relative value is large.

Active-constraint and raw physical violations must not exceed $10^{-5}$, artificial slacks must not exceed $10^{-6}$, and complementarity products must satisfy $10^{-7}$. Load shedding and PV curtailment must each be at most 0.10 MWh over the four unweighted representative days. For final verification, all artificial slacks are fixed to zero and the placement is solved again. The conic inequality remains in place, and the same audit is repeated. Throughout this paper, “hard-verified” means that this procedure passed; it does not denote a separate AC power-flow reconstruction or a proof of relaxation exactness.

## 3.4. Operational sequence

1. Build and verify the base operating case, and establish its annual economic reference cost.
2. Initialize the neural network and generate a placement trajectory from the empty state using legal-action MCTS.
3. Solve and audit the terminal placement. Record its cost, status, and diagnostics.
4. Add trajectories with usable numerical outcomes to replay and update the network. Retain solved noncompliant trajectories as negative samples; exclude numerical failures.
5. Repeat for the prescribed training episodes while preserving the best recorded compliant placement.
6. Generate a separate trajectory from the final checkpoint without root noise and with deterministic visit-count action selection.
7. Re-solve selected designs with zero artificial slacks, repeat the audit, and report economic results separately for each selection protocol.

# 4. Experimental protocol

## 4.1. Main training and checkpoint experiments

The main experiment comprises 400 training episodes from random initialization. It yields both a final network checkpoint and a historical collection of evaluated placements. Two subsequent design-selection protocols are kept separate. The **direct checkpoint** protocol starts from the empty placement and uses 500 simulations per decision, zero-temperature action selection, and no root noise. The **best self-play** protocol selects the lowest-cost compliant placement recorded during training and re-evaluates that placement. Both are subject to final hard verification.

**Table 2. Main computational and physical settings.**

| Setting | Value |
|---|---|
| Electrical bases | 1 MVA; 12.66 kV |
| Voltage limits / substation limit | 0.95–1.05 p.u.; 10 MVA |
| Branch limit assumption | 5-MVA rating converted to nominal-voltage current |
| Seasonal days and annual weights | Four 24-h days; 92, 93, 90, 90 |
| Training episodes / training seed | 400 / 0 |
| Training and direct-checkpoint tree simulations | 500 per executed decision |
| Maximum executed decisions | 8 |
| Replay capacity / minibatch size | 10,000 / 32 |
| Network updates | 4 per usable episode after replay reaches 32 records |
| Pilot benchmark | One search trial; cap of 8 new soft OPF evaluations |
| Pilot neural tree simulations | 100 per executed decision |
| Price sensitivity | Import/export multipliers of 0.8, 1.0, 1.2 |

## 4.2. Limited-budget benchmark

The pilot compares random search, budgeted greedy search, and a noisy multi-start neural-MCTS procedure using the final checkpoint. Each method has the same upper limit of eight new soft operating-model evaluations. Cached repeats do not consume additional evaluations. All selected candidates receive a common hard finalization outside this search budget. Initialization, search, and finalization time are reported separately in the underlying data.

Random search samples a placement length between zero and eight and then samples legal installations. Budgeted greedy search evaluates single-device additions in a seeded random order and retains the best improvement encountered before the budget is exhausted. With such a small budget, it does not exhaustively test the full neighborhood. Neural-MCTS generates multiple trajectories with root noise and deterministic visit-count selection within each trajectory, evaluates distinct terminal placements, and retains the best observed candidate. It stops at the physical-evaluation cap or after 200 trajectory attempts. This multi-start procedure is different from the single noise-free direct-checkpoint protocol.

There is one search trial with seed zero and one underlying trained model. The experiment therefore supports a descriptive comparison for a common evaluation cap. It does not equalize total computation, since tree simulations and prior training are additional resources, and it cannot support confidence intervals or significance tests.

## 4.3. Fixed-placement price sensitivity

Six placements are retained: the base case, direct checkpoint, best self-play design, and the three benchmark outputs. Both purchase and sale prices are multiplied by a common factor $\mu\in\{0.8,1.0,1.2\}$. Network data, placements, investment coefficients, gas charges, and load and PV profiles remain fixed. Each design is re-dispatched and hard-verified, giving 18 evaluations.

Savings use the base case re-dispatched under the same price factor:

$$
\operatorname{Saving}(x,\mu)=100\,
\frac{C_{\mathrm{base}}(\mu)-C_{\mathrm{econ}}(x,\mu)}{C_{\mathrm{base}}(\mu)}.
\qquad(12)
$$

The common positive multiplier preserves the purchase-to-sale price ordering; the explicit complementarity restriction is also retained. The network input contains placements but not prices, and the checkpoint is not retrained. Thus this experiment measures the response of a fixed design set under changed tariffs, rather than the generalization of a price-conditioned planning policy.

# 5. Results

## 5.1. Training behavior

Of 400 episodes, 341 produced usable numerical outcomes and 59 were excluded, giving a usable-outcome rate of 85.25%. The recorded termination categories include 54 max-iteration/time-limit outcomes and five infeasible outcomes. These exclusions should not be read as proof that 59 placements were physically infeasible; some runs ended because of numerical limits. Their exclusion can also bias the learned sample set toward placements that the operating solver handles more easily.

The run accumulated 1,348 optimizer updates and 1,940 replay records. Summed episode times were 20.23 h. Figure 2 shows the recorded annual cost and loss histories. Cost smoothing is applied to the last 20 valid samples after invalid outcomes are removed, not to a fixed block of 20 consecutive episode numbers. The cumulative best trace represents the best result recorded during training, before the selected design's later hard re-evaluation.

![Figure 2. Cost and loss histories from the single 400-episode training run. Rolling means use the most recent 20 available records and begin after five records. These are training diagnostics, not independent test-set estimates.](figures/fig2_training.png){width=6.5in}

The mean of the first 50 recorded total-loss values is 4.9232, compared with 4.3379 for the last 50, a decrease of 11.89%. There are 337 loss records in total. This decrease indicates progress on the generated training targets but does not establish convergence of the planning policy or calibration of its value predictions. The separation between the final policy result and the best recorded design provides a more direct test of that distinction.

## 5.2. Main economic outcomes

Table 3 reports the three principal designs after hard verification. The base case has an annual economic cost of USD 571,353.16. The direct checkpoint installs one gas unit at bus 22 and reduces this to USD 533,605.78, a saving of USD 37,747.38 or 6.61%.

**Table 3. Main hard-verified outcomes; bus numbers follow IEEE numbering.**

| Design-selection protocol | Gas-unit buses | Annual cost (USD) | Saving vs. base (%) |
|---|---|---:|---:|
| Base case | None | 571,353.16 | 0.00 |
| Direct checkpoint | 22 | 533,605.78 | 6.61 |
| Best recorded self-play design | 11, 14, 16, 24, 32, 33 | 403,888.92 | 29.31 |

The lowest-cost placement recorded during training occurred in episode 274. It contains six gas units, with 3 MW total installed active-power capacity, and no new storage, SVC, or capacitor bank. Its training cost was USD 403,889.00; the hard re-evaluation returned USD 403,888.92. The verified saving is USD 167,464.24 per year. Figure 3 shows the corresponding locations; the physical dispatch of each unit can vary independently over time within its capability limits.

![Figure 3. Fixed IEEE 33-bus topology with existing PV sites and the six gas installations in the best recorded training design. The drawing is schematic and not geographically scaled.](figures/fig4_placement.png){width=6.5in}

The USD 129,716.86 gap between the direct checkpoint and the best historical design is substantial. It shows that a useful design archive can coexist with a less effective final deterministic policy. Possible explanations include approximation error, differences between exploratory and deterministic trajectories, limited training diversity, and early STOP selection. These explanations are hypotheses; no ablation experiment in this study identifies their individual effects. The 29.31% result must therefore be attributed to the best design discovered during training, rather than to direct checkpoint inference.

## 5.3. Physical and numerical audit

All three principal results have zero artificial slack, zero artificial penalty cost, and zero branch-hour points that fail the combined gap test. For the six-unit design, the maximum active-constraint violation is $5.83\times10^{-13}$ and the minimum voltage is approximately 0.95000005 p.u. Representative load shedding and PV curtailment are $2.48\times10^{-7}$ MWh and $3.74\times10^{-7}$ MWh, respectively, both far below the 0.10-MWh targets.

The maximum absolute branch-flow gap is $3.65\times10^{-6}$ per-unit squared, below the absolute threshold. The maximum relative gap is about 0.9998. These two maxima need not occur at the same branch-hour point. A high relative gap can occur close to zero flow, so acceptance follows the stated pointwise absolute-or-relative rule. The data establish numerical compliance under that rule; they do not establish an exact AC solution or a globally optimal dispatch.

The base and direct-checkpoint cases also meet the performance targets despite nonzero representative load shedding of approximately 0.01149 MWh and 0.00755 MWh. Consequently, “compliant” should not be interpreted as zero unserved energy. For the direct-checkpoint case, annual economic cost includes approximately USD 3,510.28 in load-shedding cost. Reporting this component prevents economic comparisons from omitting a term simply because its associated energy is below an acceptance target.

For the best design, the numerical objective is USD 411,312.60, whereas the reported economic cost is USD 403,888.92. The difference, approximately USD 7,423.68, is the current-regularization contribution; the artificial-slack penalty is zero. This distinction is material and is retained in the accompanying result tables.

## 5.4. Pilot benchmark

Table 4 presents the single-trial benchmark. Random and greedy search each use eight new physical evaluations. Neural-MCTS uses seven, since repeated candidate trajectories exhaust the attempt limit before the eighth distinct evaluation is obtained. Its 200 attempts produce seven distinct candidates and 193 duplicates, with 44,100 tree simulations. Candidate repetition is therefore an observed limitation of this configuration.

**Table 4. Pilot results under a common cap of eight new soft OPF evaluations.**

| Search method | Gas buses | Annual cost (USD) | Saving (%) | New OPF solves | Search time (min) | Total time (min) |
|---|---|---:|---:|---:|---:|---:|
| Random | 15, 17 | 491,841.63 | 13.92 | 8 | 30.84 | 36.66 |
| Budgeted greedy | 31 | 522,415.94 | 8.57 | 8 | 18.67 | 24.92 |
| Neural-MCTS, multi-start | 22, 32 | 489,176.85 | 14.38 | 7 | 19.02 | 23.08 |

Total time includes method initialization, search, and final hard verification. It excludes the preceding 20.23-h neural training run and the separately evaluated common base case. Hardware specifications were not recorded in the experiment files; the times are descriptive measurements from the reported execution and cannot support a portable hardware-normalized speed comparison.

Neural-MCTS returns the lowest cost in this pilot, improving on random search by USD 2,664.78 per year, or approximately 0.54% of the random design's cost. Its selected two-unit design also differs from the direct checkpoint's one-unit design. That difference follows from the multi-start, noisy candidate-generation protocol and should not be hidden under a common “MCTS output” label. With one trial and unequal realized computation, the pilot provides no basis for a general superiority claim.

## 5.5. Price sensitivity

Every one of the 18 fixed-placement redispatch cases passes hard verification. Table 5 reports annual costs, while Figure 4 shows both costs and savings relative to the corresponding base. The best training design remains the cheapest of the six retained placements at each of the three tested multipliers.

**Table 5. Annual economic cost under fixed-placement price sensitivity (USD/year).**

| Fixed placement | Price multiplier 0.8 | Price multiplier 1.0 | Price multiplier 1.2 |
|---|---:|---:|---:|
| Base | 458,152.91 | 571,353.16 | 684,553.93 |
| Direct checkpoint | 438,665.97 | 533,605.78 | 626,054.06 |
| Best self-play | 372,333.51 | 403,888.92 | 425,461.03 |
| Benchmark random | 415,155.30 | 491,841.63 | 563,845.02 |
| Benchmark greedy | 429,397.72 | 522,415.94 | 612,868.39 |
| Benchmark neural-MCTS | 412,966.06 | 489,176.85 | 560,540.25 |

![Figure 4. Sensitivity of six fixed placements to simultaneous scaling of purchase and sale prices. Lines connect the three evaluated points and are not additional simulated scenarios.](figures/fig3_price_sensitivity.png){width=6.5in}

At multiplier 0.8, the best training design saves USD 85,819.40 per year, or 18.73%. At the nominal multiplier, it saves 29.31%, and at multiplier 1.2 it saves USD 259,092.91, or 37.85%. Monetary differences use unrounded source values. Gas operating charges remain fixed at USD 70/MWh in all scenarios, whereas purchase prices increase. The increasing economic advantage of this gas-rich design is consistent with that cost relationship. Since sale prices change at the same time and dispatch adjusts, the experiment does not isolate an import-price effect by itself.

# 6. Discussion and limitations

The study demonstrates a usable computational route from discrete candidate generation to an audited dispatch result. It also shows why the result should be framed as an economic case study. The base system already passes its operating targets; the main improvement is reduced modeled annual cost, accompanied by lower unserved energy in the best design. The experiment is not evidence of recovering an otherwise infeasible base network.

The predominance of gas units is strongly conditioned by the inputs. Under the adopted USD 30,000/MW investment assumption, a 0.5-MW gas unit has an upfront cost of USD 15,000 and an annualized investment cost of approximately USD 1,203.64. Its modeled operating charge is USD 70/MWh. It can also support reactive power without startup or ramp restrictions. By comparison, storage incurs investment, throughput cost, conversion losses, and a daily energy-balance requirement, while reactive-only devices do not directly replace purchased active energy. Existing PV inverter support further affects the marginal value of SVCs and capacitors. These factors offer an engineering interpretation of the discovered designs, but the absence of other technologies from the best recorded plan does not prove their general economic inferiority.

Several boundaries constrain the evidence. First, one training seed and one pilot search trial do not characterize variability. The best historical plan is selected from hundreds of training episodes, whereas the online benchmark is limited to eight physical evaluations. Their costs cannot be used as a like-for-like comparison of search efficiency. Second, the full operating model is nonconvex, and its dispatch remains influenced by current regularization. A zero-slack audit checks the accepted point rather than certifying the global minimum or exactness of the relaxation. Sensitivity to the regularization coefficient and audit thresholds has not been tested.

Third, fixed representative days omit forecast distributions, extreme operating years, and network contingencies. Several device and feeder settings are simplified, including branch ampacities, continuous capacitor control, gas commitment, and the absence of a detailed storage lifetime model. The PV profiles are used as provided, including their small nonzero spring nighttime floor; this should be revisited before applying the study to measured operating conditions. Fourth, market prices and gas charges are assumptions rather than a calibrated procurement and fuel-price dataset. The sensitivity experiment varies one common electricity-price factor and does not explore joint fuel, investment, and load uncertainty.

The most informative extensions are repeated independent training and search runs; comparisons with non-neural search and policy-only variants; stronger planning baselines under explicitly matched computation; and an independent AC feasibility reconstruction. Adding prices and load descriptors to the network input would enable a separate investigation of conditional policy generalization. Screening more diverse terminal candidates could also address the high duplicate rate seen in the pilot. These are proposed investigations, not experiments completed in the present work.

# 7. Conclusion

An implementation of reinforcement learning and neural-guided MCTS was evaluated for fixed-capacity equipment placement in the IEEE 33-bus active distribution network. A seasonal DistFlow-based operating model linked each selected placement to annual economic cost, and a separate zero-slack re-solve and numerical audit governed final acceptance. The workflow explicitly separated physical-evaluation cost, artificial penalties, current regularization, and the design-selection protocol.

The best design recorded in one 400-episode training run installed six gas units and achieved a hard-verified annual cost of USD 403,888.92, reducing the base cost by 29.31%. The direct final-checkpoint trajectory achieved a distinct 6.61% saving. Neural-MCTS obtained the lowest cost in the single-trial pilot, while repeated candidate generation limited its use of the available physical-evaluation budget. Fixed-placement redispatch showed savings of 18.73% to 37.85% for the best training design at the three tested price multipliers.

The results support the feasibility of the implemented search-and-audit workflow for this specified study. They also show that a strong historical design does not necessarily imply a comparably strong final policy. Establishing broader performance requires repeated experiments, stronger computational baselines, and additional physical and economic validation.

# Data availability

The accompanying source package contains the training log, derived evaluation tables, feeder data, representative profiles, and figure assets used in this manuscript. It does not include a complete executable planning environment or the trained checkpoint.

# References

[1] M. E. Baran and F. F. Wu, “Network reconfiguration in distribution systems for loss reduction and load balancing,” *IEEE Transactions on Power Delivery*, vol. 4, no. 2, pp. 1401–1407, 1989. DOI: [10.1109/61.25627](https://doi.org/10.1109/61.25627).

[2] S. H. Low, “Convex relaxation of optimal power flow—Part II: Exactness,” *IEEE Transactions on Control of Network Systems*, vol. 1, no. 2, pp. 177–189, 2014. DOI: [10.1109/TCNS.2014.2323634](https://doi.org/10.1109/TCNS.2014.2323634).

[3] C. B. Browne et al., “A survey of Monte Carlo tree search methods,” *IEEE Transactions on Computational Intelligence and AI in Games*, vol. 4, no. 1, pp. 1–43, 2012. DOI: [10.1109/TCIAIG.2012.2186810](https://doi.org/10.1109/TCIAIG.2012.2186810).

[4] D. Silver et al., “A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play,” *Science*, vol. 362, no. 6419, pp. 1140–1144, 2018. DOI: [10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404).

[5] X. Zhang, W. Hua, Y. Liu, J. Duan, Z. Tang, and J. Liu, “Reinforcement learning for active distribution network planning based on Monte Carlo tree search,” *International Journal of Electrical Power & Energy Systems*, vol. 138, article 107885, 2022. DOI: [10.1016/j.ijepes.2021.107885](https://doi.org/10.1016/j.ijepes.2021.107885).

[6] N. Li and Y. Gao, “Real-time pricing based on convex hull method for smart grid with multiple generating units,” *Energy*, vol. 285, article 129543, 2023. DOI: [10.1016/j.energy.2023.129543](https://doi.org/10.1016/j.energy.2023.129543).

[7] A. Wächter and L. T. Biegler, “On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming,” *Mathematical Programming*, vol. 106, no. 1, pp. 25–57, 2006. DOI: [10.1007/s10107-004-0559-y](https://doi.org/10.1007/s10107-004-0559-y).

[8] A. Y. Ng, D. Harada, and S. J. Russell, “Policy invariance under reward transformations: Theory and application to reward shaping,” in *Proceedings of the 16th International Conference on Machine Learning*, 1999, pp. 278–287.
