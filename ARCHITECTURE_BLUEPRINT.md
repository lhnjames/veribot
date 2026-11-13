# VeriBot: Bidirectional Verifiable Social Bot Detection Blueprint

## 1. Simplified Paper-Ready Nomenclature

To eliminate idiosyncratic terminology and align with top-tier publication standards (NeurIPS, ICLR, KDD), all system components, models, and protocols are standardized as follows:

| Legacy Term | Standard Academic Name | Math Symbol | Core Function / Responsibility |
| :--- | :--- | :---: | :--- |
| `CleanCoalitionNet` (Model A) | **Forward Evidence Miner** | $M_{\text{fwd}}$ | Identifies sparse, verifiable positive evidence subgraphs $E^+_{\text{fwd}}$ sufficient to justify a Bot decision. |
| `CleanChallengerNet` (Model B) | **Heterogeneous Challenger** | $M_{\text{rev}}$ | Independent Set-Transformer adversary; mines positive witnesses $E^+_{\text{rev}}$ and human counterexamples $E^-_{\text{rev}}$. |
| `PayloadGroundedEncoder` | **Zero-Preserving Encoder** | $\Phi_{\text{zero}}$ | Maps raw multi-modal features $X$ to embeddings $H$ such that zero-payload inputs guarantee zero hidden activation: $\Phi(0) \equiv 0$. |
| `Tier 1 Minimal Coalition` | **Minimal Sufficient Coalition** | $C^*$ | The smallest source-closed evidence subset $C^* \subseteq E^+_{\text{fwd}}$ whose features alone maintain $M_{\text{fwd}}(C^*) \ge \tau$. |
| `Tier 2 Conditional Flip` | **Counterfactual Necessity** | $\Delta_{\text{CF}}$ | The logit drop when $C^*$ is ablated: $M_{\text{fwd}}(E^+ \setminus C^*) < \tau$. |
| `Tier 3 Or-of-Ands Witness` | **Disjoint Witness Decomposition** | $\mathcal{W}_{\text{disj}}$ | Iterative extraction of mutually disjoint evidence chains $\{C_1^*, C_2^*, \dots\}$ in the residual graph $V \setminus E^+_{\text{fwd}}$. |
| `Cross-Support Agreement` | **Bidirectional Consensus** | $M_{\text{fwd}} \rightleftharpoons M_{\text{rev}}$ | Mutual verification: $M_{\text{rev}}(E^+_{\text{fwd}}) \ge \tau_{\text{rev}}$ and $M_{\text{fwd}}(E^+_{\text{rev}}) \ge \tau_{\text{fwd}}$. |
| `Calibrated Refutation` | **Counterexample Arbitration** | $\mathcal{R}_{\text{cal}}$ | Overrules $M_{\text{fwd}}$ false positives when $M_{\text{rev}}$ discovers statistically calibrated human counterevidence $E^-_{\text{rev}}$. |

---

## 2. Complete Architectural Design

```mermaid
flowchart TD
    subgraph Input["Multi-Modal Input Account u"]
        P["Payload Features P<br/>(Profile, Tweets, Neighbors, Text Embeddings)"]
        T["Type Indicators T<br/>(Static Group Family & Direction Flags)"]
    end

    subgraph Encoders["Zero-Preserving Encoders"]
        EncA["Phi_zero (Forward Miner)<br/>Linear(P) + RMSNorm + TypeFiLM(T)"]
        EncB["Phi_zero (Heterogeneous Challenger)<br/>Linear(P) + RMSNorm + TypeFiLM(T)"]
    end

    P --> EncA
    T --> EncA
    P --> EncB
    T --> EncB

    subgraph ForwardMiner["Forward Evidence Miner (M_fwd)"]
        GateA["Local Unbiased Gate<br/>Linear(H_local, GateProj(T)) -> pi_fwd"]
        PoolA["Intra-Family Normalized Pooling"]
        BiA["Cross-Family Bilinear Synergy Head<br/>H_family_i x H_family_j"]
        OutA["Logit Scorer + Global Bias b_0<br/>Output: Logit_fwd, Evidence E+_fwd"]
    end

    EncA --> GateA
    GateA --> PoolA
    PoolA --> BiA
    BiA --> OutA

    subgraph Challenger["Heterogeneous Challenger (M_rev)"]
        STB["2-Layer Set Transformer<br/>Multi-Head Self-Attention over Groups"]
        QBot["Cross-Attention Query q_bot<br/>Attends over Bot Witnesses E+_rev"]
        QHum["Cross-Attention Query q_human<br/>Attends over Human Counterexamples E-_rev"]
        OutB["Output: Logit_rev, E+_rev, E-_rev"]
    end

    EncB --> STB
    STB --> QBot
    STB --> QHum
    QBot --> OutB
    QHum --> OutB

    subgraph VerificationEngine["Bidirectional Verification & Arbitration Engine"]
        T1["Protocol 1: Minimal Sufficient Coalition C*<br/>Greedy Source-Closed Reduction"]
        T2["Protocol 2: Counterfactual Necessity Delta_CF<br/>Source-Closed Ablation vs Matched Random"]
        T3["Protocol 3: Disjoint Witness Mining<br/>Search in V \ E+_fwd (Residual Evidence)"]
        T4["Protocol 4: Counterexample Arbitration<br/>Arbitrate FP with E-_rev via Calibrated Thresholds"]
    end

    OutA --> T1
    T1 --> T2
    T2 --> T3
    OutB --> T4
    T3 --> T4

    T4 --> Decision["Final Decision & Verifiable Audit Certificate"]
```

---

## 3. Mathematical Foundations & Invariants

### 3.1 Mathematical Zero-Payload Invariance (Zero-Leakage)
To guarantee that unselected or zero-payload groups transmit zero discriminative signal:
1. **Bias-Free Projection**: All feature transformation layers omit bias terms:
   $$H_{\text{local}} = \text{RMSNorm}(W_{\text{pay}} P), \quad W_{\text{pay}} \in \mathbb{R}^{d \times d_p}, \text{bias} = \text{False}$$
2. **Type Modulation**: Group type indicators $T$ modulate scale only, preserving $P = 0 \implies H_{\text{local}} = 0$:
   $$H = H_{\text{local}} \odot \left(1 + 0.5 \tanh(W_{\text{type}} T + b_{\text{type}})\right)$$
3. **Normalized Pooling**: Pooled representations normalize strictly by active count $\max(\sum m_i, 1)$, preventing group cardinality from acting as an implicit shortcut.
4. **Exact Zero Invariant**:
   $$\forall m \in \{0, 1\}^M, \quad M_{\text{fwd}}(0, m) = b_0, \quad M_{\text{rev}}(0, m) = b_0$$
   Consequently, $\text{Var}_{m \sim \mathcal{B}}(M(0, m)) \equiv 0.0$ and $\text{AUC}_{\text{mask-only}} \equiv 0.5000$.

### 3.2 Formal Protocols of Bidirectional Verification

#### Protocol 1: Minimal Sufficient Coalition ($C^*$)
Given predicted evidence $E^+_{\text{fwd}} = \{g_1, \dots, g_m\}$ such that $M_{\text{fwd}}(E^+_{\text{fwd}}) \ge \tau$:
$$C^* = \arg\min_{C \subseteq E^+_{\text{fwd}}} |C| \quad \text{s.t.} \quad M_{\text{fwd}}(C) \ge \tau \quad \text{and} \quad \text{SourceClosed}(C)$$
- **Sufficiency**: $M_{\text{fwd}}(C^*) \ge \tau$
- **Singleton Irreducibility**: $\forall g \in C^*, M_{\text{fwd}}(\{g\}) < \tau$
- **Synergy Gain**: $\text{Syn}(C^*) = M_{\text{fwd}}(C^*) - \sum_{g \in C^*} M_{\text{fwd}}(\{g\})$

#### Protocol 2: Counterfactual Necessity ($\Delta_{\text{CF}}$)
Ablating $C^*$ along with all co-dependent raw source items:
$$\Delta_{\text{CF}} = M_{\text{fwd}}(E^+_{\text{fwd}}) - M_{\text{fwd}}(E^+_{\text{fwd}} \setminus \text{SourceClosure}(C^*))$$
Verified against matched-size random controls:
$$\Delta_{\text{CF}} > \text{Median}\left(\{\Delta_{\text{random}}^{(r)}\}_{r=1}^R\right) \quad \text{and} \quad M_{\text{fwd}}(E^+_{\text{fwd}} \setminus C^*) < \tau$$

#### Protocol 3: Disjoint Witness Decomposition ($\mathcal{W}_{\text{disj}}$)
In the residual subgraph $V \setminus \text{SourceClosure}(C^*)$, search iteratively for alternative witness sets:
$$C_k^* = \text{ExtractCoalition}(V \setminus \bigcup_{j=1}^{k-1} C_j^*)$$
Demonstrates whether account exhibits single-chain vulnerability or structural multi-chain resilience.

#### Protocol 4: Counterexample Arbitration ($\mathcal{R}_{\text{cal}}$)
Given $M_{\text{fwd}}(u) \ge \tau_{\text{fwd}}$, inspect challenger counterevidence $E^-_{\text{rev}}$:
$$\text{Decision}(u) = \begin{cases} 
0 (\text{Human}), & \text{if } M_{\text{fwd}}(u) \le \tau_{\text{cap}} \text{ and } \text{Score}_{\text{human}}(E^-_{\text{rev}}) \ge \theta_{\text{p95}} \\
1 (\text{Bot}), & \text{otherwise}
\end{cases}$$
Where $\theta_{\text{p95}}$ is strictly calibrated on a 50% held-out Human development split.

---

## 4. Comprehensive Parameter & Configuration Table

| Category | Hyperparameter | Symbol | Value | Tuning & Selection Protocol |
| :--- | :--- | :---: | :---: | :--- |
| **Model Dimensions** | Hidden Dimension | $d_h$ | `128` | Standard capacity; grid tested $\{64, 128, 256\}$ on Dev. |
| | Bilinear Pair Dimension | $d_{\text{pair}}$ | `32` | Compressed rank for cross-family outer product. |
| | Number of Attention Heads | $h$ | `4` | 32 dimensions per head for Set Transformer. |
| | Transformer Layers | $L$ | `2` | Sufficient for group-level topological interaction. |
| | Query Heads in $M_{\text{fwd}}$ | $K_q$ | `2` | Dual query multi-head pooling. |
| **Optimization** | Optimizer | - | `AdamW` | Standard decoupled weight decay. |
| | Peak Learning Rate | $\eta$ | `6e-4` | Linearly decayed; selected from $\{3e-4, 6e-4, 1e-3\}$. |
| | Weight Decay | $\lambda_{\text{wd}}$ | `2e-4` | $L_2$ regularization on non-bias parameters. |
| | Batch Size | $B$ | `32` | Stable gradient estimation across variable-group accounts. |
| | Dropout Rate | $p$ | `0.15` | Applied before residual additions and output projections. |
| | Gradient Clipping | $\|\nabla\|$ | `1.0` | Global gradient norm threshold. |
| **Evidence Selection** | Minimum Family Fraction | $\rho_{\min}$ | `0.24` | Lower bound on group retention per family. |
| | Maximum Family Fraction | $\rho_{\max}$ | `0.38` | Upper bound on group retention per family. |
| | Budget Penalty Weight | $\lambda_{\text{budget}}$ | `2.5` | Smooth $L_1$ penalty outside $[\rho_{\min}, \rho_{\max}]$. |
| | Family Drop Rate | $p_{\text{drop}}$ | `0.25` | Random modality dropout during training to prevent family starvation. |
| | Subset Stability Weight | $\lambda_{\text{sub}}$ | `0.40` | Encourages consistency on random 55%-85% subsets of $E^+$. |
| | Logit $L_2$ Regularization | $\lambda_{\text{l2}}$ | `0.004` | Penalizes extreme overconfident logit explosions. |
| **Inference & Auditing** | Decision Threshold | $\tau$ | Dev-optimal | Vectorized search maximizing Macro-F1 strictly on Dev. |
| | Random Control Trials | $R$ | `10` | Matched-cardinality random subsets per audited account. |
| | Disjoint Search Depth | $K_{\max}$ | `4` | Maximum Or-of-Ands recursion depth. |
| | Multi-Probe Search Budget| $Q$ | `4` | Prefix top-k budget for residual witness extraction. |

---

## 5. System Execution & State Transition Diagrams

### 5.1 Bidirectional Verification Protocol (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    actor Auditor as User / Regulator
    participant Miner as Forward Miner (M_fwd)
    participant Challenger as Heterogeneous Challenger (M_rev)
    participant Engine as Verification Engine
    participant Registry as Audit Log / Certificate

    Auditor->>Miner: Submit target account graph G_u
    Miner->>Miner: Forward inference with Unbiased Gate
    Miner-->>Engine: Candidate evidence E+_fwd, Logit_fwd
    
    alt Logit_fwd < Threshold_fwd
        Engine-->>Auditor: Return Decision: HUMAN (Evidence gate closed)
    else Logit_fwd >= Threshold_fwd
        Engine->>Engine: Protocol 1: Compute Minimal Coalition C*
        Engine->>Engine: Protocol 2: Compute Counterfactual Flip Delta_CF
        Engine->>Challenger: Request independent witness & challenge on V \ C*
        Challenger->>Challenger: Set-Transformer multi-head cross-attention
        Challenger-->>Engine: Witness E+_rev, Human Counterexample E-_rev, Logit_rev
        
        Engine->>Engine: Mutual Cross-Evaluation: M_rev(E+_fwd) & M_fwd(E+_rev)
        
        alt E-_rev exceeds Calibrated Threshold theta_p95
            Engine->>Engine: Protocol 4: Refute False Positive (Camouflage Detected)
            Engine-->>Registry: Log Refuted Account & Counterexample Subgraph
            Engine-->>Auditor: Return Decision: HUMAN (Refuted by Challenger)
        else Bidirectional Consensus Confirmed
            Engine->>Engine: Protocol 3: Compute Disjoint Witness Depth
            Engine-->>Registry: Issue Verifiable Multi-Tier Certificate
            Engine-->>Auditor: Return Decision: BOT (Certified with C*, Delta_CF, W_disj)
        end
    end
```

### 5.2 Micro-Level Account Verification State Machine

```mermaid
stateDiagram-v2
    [*] --> Uninspected: Account Received

    state Uninspected {
        [*] --> FeatureIngestion
        FeatureIngestion --> ZeroPreservingEncoding: Bias-Free Mapping
    }

    Uninspected --> CandidateBot: M_fwd(E+) >= tau_fwd
    Uninspected --> CertifiedHuman: M_fwd(E+) < tau_fwd

    state CandidateBot {
        [*] --> CoalitionReduction: Greedy Source-Closed Reduction
        CoalitionReduction --> CounterfactualTesting: Ablate C* vs Random
        CounterfactualTesting --> ChallengerInterrogation: Probe Residual V \ C*
    }

    CandidateBot --> RefutedHuman: Challenger finds Valid E- (Refuted FP)
    CandidateBot --> CertifiedBot: Bidirectional Consensus Achieved

    state CertifiedBot {
        [*] --> SingleChainWitness: Disjoint Depth K = 1
        [*] --> MultiChainResilient: Disjoint Depth K >= 2
    }

    RefutedHuman --> [*]: Audited Output
    CertifiedHuman --> [*]: Audited Output
    CertifiedBot --> [*]: Audited Output
```

---

## 6. Micro-Level Diagnosis & Algorithmic Refinement Taxonomy

Every evaluated account is comprehensively categorized into one of five mutually exclusive operational states with principled algorithmic remediations:

| Diagnostic State | Observable Symptoms | Root Cause Analysis | Formal Algorithmic Remediation |
| :--- | :--- | :--- | :--- |
| **True Positive (Certified Bot)** | $M_{\text{fwd}} \ge \tau$, $M_{\text{rev}} \ge \tau$, $M_{\text{rev}}(E^+_{\text{fwd}}) \ge \tau$. Mutual consensus holds; $C^*$ is irreducible. | Strong multi-modal coordination across profile, relations, and text groups. | Maintain sparse budget $[\rho_{\min}, \rho_{\max}]$; reinforce cross-family bilinear synergy margins. |
| **True Negative (Certified Human)** | $M_{\text{fwd}} < \tau$, unselected gates closed. Zero evidence coalition can be constructed. | Natural human account without coordinated repetitive behavioral patterns. | Maintain singleton humility penalty to prevent accidental fires on isolated groups. |
| **Refuted False Positive (Challenger Win)** | $M_{\text{fwd}} \ge \tau$ initially, but $M_{\text{rev}}$ identifies strong $E^-_{\text{rev}}$ in background groups with score $\ge \theta_{\text{p95}}$. | Forward miner suffered from confirmation bias on isolated tweet spikes; Challenger found authentic human profile and organic social dialogue. | Counterexample arbitration successfully overturns the decision, reducing false alarm rate without degrading bot recall. |
| **False Positive (Unrefuted Alarm)** | $M_{\text{fwd}} \ge \tau$, and Challenger failed to overturn ($E^-_{\text{rev}} < \theta$). | Account has sparse profile metadata or zero active graph connections, depriving Challenger of candidate counter-evidence groups. | **Algorithmic Path**: Introduce adaptive prior baseline dependent on group cardinality $|V_u|$; penalize low-entropy textual groups. |
| **False Negative (Stealth Bot)** | $M_{\text{fwd}} < \tau$; bot behavior fell below the decision boundary. | Advanced camouflage (e.g. human-mimicking text generation, stealthy follow-unfollow dynamics). | **Algorithmic Path**: Expand contextual graph receptive field via multi-hop motif aggregation; train contrastive challenger specifically on borderline logits. |

