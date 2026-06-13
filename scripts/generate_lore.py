"""Generate synthetic D&D lore and data-science concept text for the corpus.

Two generators are included:

  lore      — Monster stat blocks, spell descriptions, magic item entries,
               location writeups, NPC profiles, and tavern encounters.
               Mirrors the vocabulary and structure of D&D 5e sourcebooks.

  datascience — Concept explanations in a textbook / encyclopedia style:
               algorithms, statistical concepts, ML methods, maths topics.

Both generators write deterministic files so re-running is idempotent: each
seed produces the same content and is skipped on subsequent runs.

Usage
-----
    python scripts/generate_lore.py
    python scripts/generate_lore.py --group lore --count 3000
    python scripts/generate_lore.py --group datascience --count 2000
    python scripts/generate_lore.py --seed 999 --output data/corpus/saga/

Output
------
Two files are written (one per group unless --group is specified):
    data/corpus/saga/synth_lore.txt
    data/corpus/saga/synth_datascience.txt

If the files already exist they are overwritten only when --force is passed.
"""

import argparse
import random
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _a(noun: str) -> str:
    return "An" if noun[0].lower() in "aeiou" else "A"


def _join_and(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]


# ---------------------------------------------------------------------------
# D&D Lore generator
# ---------------------------------------------------------------------------

_CREATURE_TYPES = [
    "aberration", "beast", "celestial", "construct", "dragon", "elemental",
    "fey", "fiend", "giant", "humanoid", "monstrosity", "ooze", "plant",
    "undead",
]

_SIZES = ["Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan"]

_ALIGNMENTS = [
    "lawful good", "neutral good", "chaotic good",
    "lawful neutral", "true neutral", "chaotic neutral",
    "lawful evil", "neutral evil", "chaotic evil",
    "unaligned",
]

_DAMAGE_TYPES = [
    "acid", "bludgeoning", "cold", "fire", "force", "lightning", "necrotic",
    "piercing", "poison", "psychic", "radiant", "slashing", "thunder",
]

_CONDITIONS = [
    "blinded", "charmed", "deafened", "exhaustion", "frightened", "grappled",
    "incapacitated", "invisible", "paralyzed", "petrified", "poisoned",
    "prone", "restrained", "stunned", "unconscious",
]

_HABITATS = [
    "dense forest", "fetid swamp", "high mountain pass", "arid desert",
    "frozen tundra", "dark dungeon", "sunken ruin", "volcanic cave",
    "feywild glade", "underdark cavern", "coastal cliff", "open plain",
    "ancient graveyard", "planar rift", "enchanted tower",
]

_CREATURE_ADJECTIVES = [
    "ancient", "twisted", "luminous", "shadow-wreathed", "arcane",
    "diseased", "spectral", "crystalline", "iron-clad", "ethereal",
    "venomous", "frost-covered", "flame-wreathed", "obsidian-skinned",
    "bone-white",
]

_CREATURE_NAMES = [
    "Vorthak", "Glaivespawn", "Shade-Render", "Bone Crawler", "Mire Drake",
    "Null Wraith", "Ember Stalker", "Frost Fiend", "Plague Asp",
    "Void Shambler", "Lunar Hound", "Tide Lurker", "Dusk Revenant",
    "Iron Golem", "Spore Walker", "Cinder Troll", "Deepwater Horror",
    "Thornback Basilisk", "Wailing Specter", "Silt Drake",
    "Ashen Chimera", "Hollow Knight", "Blight Toad", "Rift Spawn",
    "Pale Watcher",
]

_CREATURE_ABILITIES = [
    "Multiattack. The creature makes two attacks on its turn.",
    "Pack Tactics. The creature has advantage on attack rolls against a creature if at least one of its allies is adjacent to that creature.",
    "Magic Resistance. The creature has advantage on saving throws against spells and other magical effects.",
    "Innate Spellcasting. The creature can innately cast spells, requiring no material components.",
    "Regeneration. The creature regains hit points at the start of its turn unless it has taken certain damage types.",
    "Ethereal Sight. The creature can see into the Ethereal Plane out to 60 feet.",
    "Spellcasting. The creature is a spellcaster with a spell save DC and spell attack bonus.",
    "Legendary Resistance. If the creature fails a saving throw it can choose to succeed instead.",
    "Aura of Fear. Creatures within 30 feet that can see the creature must succeed on a Wisdom saving throw or become frightened.",
    "Death Throes. When reduced to 0 hit points the creature explodes in a burst of energy.",
    "Keen Senses. The creature has advantage on Perception checks relying on sight, hearing, or smell.",
    "Amphibious. The creature can breathe air and water.",
    "False Appearance. While motionless, the creature is indistinguishable from a mundane object.",
    "Damage Absorption. When the creature takes certain damage it instead regains hit points.",
    "Trampling Charge. If the creature moves at least 20 feet toward a target and hits with a horn or claw attack, the target must make a Strength saving throw or be knocked prone.",
]

_SPELL_SCHOOLS = [
    "Abjuration", "Conjuration", "Divination", "Enchantment",
    "Evocation", "Illusion", "Necromancy", "Transmutation",
]

_SPELL_NAMES = [
    "Ashen Veil", "Bone Prison", "Caustic Wave", "Dream Snare",
    "Ember Crown", "Fate's Thread", "Glacial Lance", "Hollow Flame",
    "Iron Shroud", "Jinx Bolt", "Killing Frost", "Luminous Ward",
    "Mind Fetter", "Null Field", "Oblique Strike", "Phantom Bulwark",
    "Quill Storm", "Rune Cage", "Shadow Grasp", "Temporal Rift",
    "Umbral Step", "Voidheart", "Warding Glyph", "Xeric Blast",
    "Yearning Hex", "Zealot's Rebuke",
]

_ITEM_TYPES = [
    "longsword", "shortsword", "dagger", "staff", "wand", "ring", "amulet",
    "cloak", "helm", "gauntlets", "boots", "shield", "bow", "tome",
    "orb", "rod",
]

_ITEM_ADJECTIVES = [
    "Sundering", "Whispering", "Eternal", "Accursed", "Radiant",
    "Hungering", "Forsaken", "Ancient", "Voidborn", "Shattered",
    "Unyielding", "Gilded", "Storm-forged", "Hollow", "Spectral",
]

_LOCATION_TYPES = [
    "dungeon", "fortress", "ruin", "tower", "cavern", "temple", "shrine",
    "keep", "crypt", "academy", "enclave", "sanctum", "monument", "vault",
]

_LOCATION_ADJECTIVES = [
    "Sunken", "Forgotten", "Shattered", "Cursed", "Ancient", "Infernal",
    "Celestial", "Verdant", "Frozen", "Ashen", "Spectral", "Gilded",
    "Blighted", "Radiant", "Hollow",
]


def _gen_monster(rng: random.Random) -> str:
    name = rng.choice(_CREATURE_NAMES)
    adj  = rng.choice(_CREATURE_ADJECTIVES)
    size = rng.choice(_SIZES)
    typ  = rng.choice(_CREATURE_TYPES)
    align = rng.choice(_ALIGNMENTS)
    ac = rng.randint(10, 22)
    hp = rng.randint(20, 300)
    hd_n = rng.randint(2, 20)
    hd_d = rng.choice([6, 8, 10, 12])
    speed = rng.choice([20, 30, 40, 50, 60])
    cr = rng.choice(["1/8", "1/4", "1/2", "1", "2", "3", "4", "5",
                     "6", "7", "8", "9", "10", "12", "13", "15", "17", "20"])
    habitat = rng.choice(_HABITATS)

    stats = {k: rng.randint(6, 24) for k in ["STR", "DEX", "CON", "INT", "WIS", "CHA"]}
    def mod(v: int) -> str:
        m = (v - 10) // 2
        return f"+{m}" if m >= 0 else str(m)

    imm = rng.sample(_damage_types_or_cond(rng), rng.randint(0, 3))
    res = rng.sample(_damage_types_or_cond(rng), rng.randint(0, 3))
    abilities = rng.sample(_CREATURE_ABILITIES, rng.randint(1, 4))

    attack_name = rng.choice(["Claw", "Bite", "Slam", "Tail Swipe", "Tentacle", "Tendril", "Gore"])
    atk_bonus = rng.randint(2, 12)
    dmg_dice = f"{rng.randint(1,3)}d{rng.choice([4,6,8,10,12])}"
    dmg_mod = rng.randint(0, 8)
    dmg_type = rng.choice(_DAMAGE_TYPES)

    lines = [
        f"## {_cap(adj)} {name}",
        f"",
        f"*{size} {typ}, {align}*",
        f"",
        f"**Armor Class** {ac}",
        f"**Hit Points** {hp} ({hd_n}d{hd_d})",
        f"**Speed** {speed} ft.",
        f"",
        "| STR | DEX | CON | INT | WIS | CHA |",
        "|-----|-----|-----|-----|-----|-----|",
        "| " + " | ".join(f"{stats[k]} ({mod(stats[k])})" for k in ["STR","DEX","CON","INT","WIS","CHA"]) + " |",
        "",
    ]
    if imm:
        lines.append(f"**Damage Immunities** {', '.join(imm)}")
    if res:
        lines.append(f"**Damage Resistances** {', '.join(res)}")
    lines += [
        f"**Challenge** {cr}",
        "",
        "### Traits",
        "",
    ]
    for ab in abilities:
        lines.append(ab)
        lines.append("")
    lines += [
        "### Actions",
        "",
        f"**{attack_name}.** *Melee Weapon Attack:* +{atk_bonus} to hit, reach 5 ft., one target. "
        f"*Hit:* {dmg_dice} + {dmg_mod} {dmg_type} damage.",
        "",
        f"The {_cap(adj)} {name} dwells in {habitat}s, where it has terrorized "
        f"communities for generations. Adventurers who face this creature should prepare "
        f"carefully, for its {rng.choice(['cunning','ferocity','resilience','arcane power'])} "
        f"is formidable.",
    ]
    return "\n".join(lines)


def _damage_types_or_cond(rng: random.Random) -> list[str]:
    return _DAMAGE_TYPES


def _gen_spell(rng: random.Random) -> str:
    name = rng.choice(_SPELL_NAMES)
    school = rng.choice(_SPELL_SCHOOLS)
    level = rng.randint(0, 9)
    level_str = "Cantrip" if level == 0 else f"{level}{'st' if level==1 else 'nd' if level==2 else 'rd' if level==3 else 'th'}-level"
    casting = rng.choice(["1 action", "1 bonus action", "1 reaction", "1 minute", "10 minutes"])
    rnge = rng.choice(["Self", "Touch", "30 feet", "60 feet", "90 feet", "120 feet", "300 feet", "1 mile"])
    components = ", ".join(rng.sample(["V", "S", "M"], rng.randint(1, 3)))
    duration = rng.choice([
        "Instantaneous", "1 round", "1 minute", "10 minutes", "1 hour",
        "8 hours", "24 hours", "Concentration, up to 1 minute",
        "Concentration, up to 10 minutes", "Concentration, up to 1 hour",
        "Until dispelled",
    ])
    classes = rng.sample(["Bard", "Cleric", "Druid", "Paladin", "Ranger",
                           "Sorcerer", "Warlock", "Wizard"], rng.randint(1, 4))
    save_stat = rng.choice(["Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma"])
    damage_type = rng.choice(_DAMAGE_TYPES)
    dice = f"{rng.randint(1,8)}d{rng.choice([4,6,8,10,12])}"
    higher_dice = f"{rng.randint(1,3)}d{rng.choice([4,6,8])}"
    effect_verb = rng.choice([
        "conjures", "channels", "shapes", "weaves", "unleashes", "summons",
        "binds", "shatters", "dissolves", "ignites",
    ])
    target_desc = rng.choice([
        "one creature you can see within range",
        "all creatures in a 20-foot radius sphere",
        "a line 60 feet long and 5 feet wide",
        "one object or surface you touch",
        "all hostile creatures within 30 feet of you",
        "a 15-foot cone",
    ])

    lines = [
        f"## {name}",
        f"",
        f"*{level_str} {school}*",
        f"",
        f"**Casting Time:** {casting}",
        f"**Range:** {rnge}",
        f"**Components:** {components}",
        f"**Duration:** {duration}",
        f"**Classes:** {', '.join(classes)}",
        f"",
        f"The caster {effect_verb} raw magical force, targeting {target_desc}. "
        f"The target must make a {save_stat} saving throw. On a failed save, "
        f"the target takes {dice} {damage_type} damage, or half as much on a success.",
        f"",
        f"Additional effects may include the {rng.choice(_CONDITIONS)} condition for "
        f"{rng.choice(['1 round', 'until the end of your next turn', '1 minute'])}.",
    ]
    if level > 0:
        lines += [
            f"",
            f"**At Higher Levels.** When you cast this spell using a spell slot of "
            f"{level + 1}th level or higher, the damage increases by {higher_dice} "
            f"for each slot level above {level}th.",
        ]
    return "\n".join(lines)


def _gen_magic_item(rng: random.Random) -> str:
    adj  = rng.choice(_ITEM_ADJECTIVES)
    typ  = rng.choice(_ITEM_TYPES)
    name = f"{adj} {_cap(typ)}"
    rarity = rng.choice(["Common", "Uncommon", "Rare", "Very Rare", "Legendary", "Artifact"])
    attunement = rng.choice([
        "", "(requires attunement)", "(requires attunement by a spellcaster)",
        "(requires attunement by a wizard)", "(requires attunement by a cleric or paladin)",
    ])
    bonus = rng.choice(["+1", "+2", "+3"])
    damage_type = rng.choice(_DAMAGE_TYPES)
    condition = rng.choice(_CONDITIONS)
    save_stat = rng.choice(["Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma"])
    dc = rng.randint(12, 20)
    charges = rng.randint(3, 10)

    lines = [
        f"## {name}",
        f"",
        f"*{typ.capitalize()}, {rarity}{' ' + attunement if attunement else ''}*",
        f"",
        f"This {typ}, crafted in an age of myth, radiates a faint magical aura. "
        f"It grants a {bonus} bonus to attack rolls and damage rolls made with it.",
        f"",
        f"**Special Property.** When you hit a creature with this {typ}, the target "
        f"must succeed on a DC {dc} {save_stat} saving throw or become {condition} "
        f"until the end of its next turn. On a hit the weapon also deals an additional "
        f"1d{rng.choice([4,6,8,10])} {damage_type} damage.",
        f"",
        f"**Charges.** The {typ} has {charges} charges and regains 1d{rng.choice([4,6])} "
        f"expended charges daily at dawn. You can expend 1 charge as a bonus action to "
        f"activate a secondary effect chosen by the DM.",
    ]
    return "\n".join(lines)


def _gen_location(rng: random.Random) -> str:
    adj  = rng.choice(_LOCATION_ADJECTIVES)
    typ  = rng.choice(_LOCATION_TYPES)
    name = f"The {adj} {_cap(typ)}"
    creature = rng.choice(_CREATURE_NAMES)
    habitat  = rng.choice(_HABITATS)
    treasure = rng.choice([
        "ancient gold coins", "a cache of rare gems", "forgotten scrolls",
        "a legendary weapon", "alchemical components", "a planar portal shard",
        "enchanted armor", "divine relics",
    ])
    history = rng.choice([
        "built by a long-dead empire",
        "consecrated to a forgotten deity",
        "raised in a single night by unknown magic",
        "said to drift between planes on moonless nights",
        "once the lair of a slain dragon",
        "constructed from the bones of a titan",
    ])
    lines = [
        f"## {name}",
        f"",
        f"Located in a {habitat}, {name} is {history}. Locals speak of it only in "
        f"hushed tones, warning travelers to keep their distance after dark.",
        f"",
        f"**Inhabitants.** The site is now home to a colony of {creature}s, who guard "
        f"its depths jealously. Several traps left by the original builders remain "
        f"functional, posing additional hazards to explorers.",
        f"",
        f"**Treasure.** Deeper within, {treasure} await those bold enough to claim them.",
        f"",
        f"**Adventure Hooks.**",
        f"- A desperate merchant offers a reward to anyone who can recover a family "
        f"heirloom lost inside.",
        f"- Strange lights have been seen emanating from the {typ} at midnight.",
        f"- A scholar believes a key to defeating a greater threat is hidden within.",
    ]
    return "\n".join(lines)


def generate_lore(rng: random.Random, count: int) -> str:
    """Generate *count* random D&D lore entries."""
    sections = []
    generators = [_gen_monster, _gen_spell, _gen_magic_item, _gen_location]
    for i in range(count):
        gen = generators[i % len(generators)]
        try:
            sections.append(gen(rng))
        except Exception:
            pass
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Data science generator
# ---------------------------------------------------------------------------

_DS_CONCEPTS = [
    ("Mean", "statistics",
     "The arithmetic mean of a dataset is the sum of all values divided by the number of observations. "
     "For a sample {x_1, x_2, ..., x_n} the mean is x̄ = (1/n) Σ x_i. "
     "The mean minimises the sum of squared deviations, a property that makes it the natural centre of a Gaussian distribution. "
     "It is sensitive to outliers: a single extreme value can shift the mean substantially, which motivates robust alternatives such as the median or trimmed mean."),

    ("Variance and Standard Deviation", "statistics",
     "Variance measures the average squared deviation from the mean: s² = (1/(n-1)) Σ (x_i - x̄)². "
     "The Bessel correction (n-1 in the denominator) makes the sample variance an unbiased estimator of the population variance. "
     "The standard deviation σ = √s² returns the spread to the original units, making it interpretable alongside the mean. "
     "A low standard deviation indicates values cluster tightly around the mean; a high one indicates a wide spread."),

    ("Normal Distribution", "probability",
     "The normal (Gaussian) distribution is characterised by its bell-shaped probability density function "
     "f(x) = (1/(σ√(2π))) exp(-(x-μ)²/(2σ²)), parameterised by mean μ and standard deviation σ. "
     "The empirical rule states that approximately 68% of observations fall within one standard deviation of the mean, "
     "95% within two, and 99.7% within three. "
     "The central limit theorem explains why the normal distribution appears so frequently: the mean of many independent random variables converges to normality regardless of the underlying distribution."),

    ("Bayes' Theorem", "probability",
     "Bayes' theorem relates the conditional and marginal probabilities of events: P(A|B) = P(B|A) P(A) / P(B). "
     "In Bayesian inference, P(A) is the prior belief, P(B|A) is the likelihood of observing data B given hypothesis A, "
     "and P(A|B) is the posterior belief updated after seeing the data. "
     "Bayesian methods allow the systematic incorporation of prior knowledge and produce full posterior distributions rather than point estimates, "
     "enabling uncertainty quantification without asymptotic approximations."),

    ("Gradient Descent", "optimisation",
     "Gradient descent is an iterative first-order optimisation algorithm that minimises a differentiable objective function f(θ). "
     "At each step the parameters are updated in the direction of the negative gradient: θ ← θ - η ∇f(θ), "
     "where η is the learning rate. "
     "Stochastic gradient descent (SGD) approximates the full gradient using a single example or mini-batch, "
     "trading exact gradient computation for computational efficiency. "
     "Adaptive methods such as Adam combine momentum and per-parameter learning rates to accelerate convergence."),

    ("Linear Regression", "machine learning",
     "Linear regression models the relationship between a response variable y and predictors X as y = Xβ + ε, "
     "where β are the coefficients and ε is a noise term assumed to be normally distributed. "
     "The ordinary least squares estimator β̂ = (XᵀX)⁻¹Xᵀy minimises the residual sum of squares. "
     "Regularised variants — Ridge regression adds an L2 penalty λ‖β‖², "
     "LASSO adds an L1 penalty λ‖β‖₁ — prevent overfitting and perform variable selection respectively."),

    ("Decision Trees", "machine learning",
     "A decision tree partitions the feature space by recursively splitting on the feature and threshold that maximally reduce impurity. "
     "Common impurity measures are the Gini coefficient G = 1 - Σ pₖ² and information gain based on entropy H = -Σ pₖ log pₖ. "
     "Trees are interpretable but prone to overfitting on training data. "
     "Random forests mitigate this by averaging many trees trained on bootstrap samples with feature subsampling, "
     "while gradient boosted trees build an ensemble sequentially, each tree correcting residuals of its predecessors."),

    ("Entropy and Information Gain", "information theory",
     "Shannon entropy H(X) = -Σ p(x) log₂ p(x) measures the average amount of information in a random variable. "
     "A uniform distribution over n outcomes has maximum entropy log₂ n bits. "
     "Kullback-Leibler divergence D_KL(P‖Q) = Σ P(x) log(P(x)/Q(x)) quantifies how much information is lost "
     "when distribution Q is used to approximate P. Mutual information I(X;Y) = H(X) - H(X|Y) measures the "
     "reduction in uncertainty about X given knowledge of Y."),

    ("Principal Component Analysis", "dimensionality reduction",
     "PCA finds a linear subspace that captures maximum variance in the data. "
     "Given a centred data matrix X ∈ ℝⁿˣᵈ, the principal components are the eigenvectors of the covariance matrix "
     "C = (1/(n-1)) XᵀX sorted by descending eigenvalue. "
     "Projecting onto the top k components preserves the most variance while reducing dimensionality from d to k. "
     "The scree plot of eigenvalues helps select k by identifying the 'elbow' where additional components contribute little."),

    ("Attention Mechanism", "deep learning",
     "The attention mechanism allows a model to dynamically weight input elements when producing each output. "
     "Scaled dot-product attention computes Attention(Q,K,V) = softmax(QKᵀ/√d_k)V, "
     "where Q, K, V are query, key, and value matrices derived from the input via learned projections. "
     "Multi-head attention runs h independent attention heads in parallel and concatenates their outputs, "
     "enabling the model to jointly attend to information from different representation subspaces. "
     "This mechanism is the core of transformer architectures."),

    ("Overfitting and Regularisation", "machine learning",
     "A model overfits when it learns the training data so well that it fails to generalise to new examples. "
     "Signs of overfitting include a large gap between training and validation loss. "
     "Regularisation techniques reduce overfitting: L2 regularisation (weight decay) penalises large weights, "
     "dropout randomly zeroes activations during training, early stopping halts training when validation loss stops improving, "
     "and data augmentation synthetically increases training set diversity."),

    ("Cross-Validation", "model evaluation",
     "K-fold cross-validation estimates model performance on unseen data by partitioning the dataset into k folds, "
     "training on k-1 folds, and evaluating on the held-out fold, repeating k times. "
     "The final estimate is the mean of k scores. Stratified k-fold preserves class proportions in each fold. "
     "Leave-one-out cross-validation (LOOCV) is the special case k=n, providing low-bias estimates at high computational cost. "
     "Nested cross-validation separates hyperparameter selection from performance estimation."),

    ("Convolutional Neural Networks", "deep learning",
     "CNNs use convolutional layers that apply learned filters across the spatial dimensions of the input. "
     "Each filter detects a local pattern (edge, texture, shape) independently of position. "
     "Max-pooling layers downsample feature maps, reducing spatial resolution while increasing translational invariance. "
     "Deep CNNs stack many convolutional and pooling layers, learning increasingly abstract representations: "
     "early layers detect edges, middle layers detect shapes, and deep layers detect object parts. "
     "Residual connections (ResNet) allow gradients to flow through very deep networks without vanishing."),

    ("Transformer Architecture", "deep learning",
     "The transformer consists of stacked encoder and decoder blocks, each containing multi-head self-attention "
     "and position-wise feed-forward sub-layers with residual connections and layer normalisation. "
     "Positional encodings (sinusoidal or learned) inject sequence order into the attention mechanism, "
     "which is otherwise permutation-invariant. "
     "Decoder-only transformers (GPT-style) mask future tokens to enable autoregressive language modelling. "
     "Scaling laws show that loss decreases predictably as model size, dataset size, and compute increase."),

    ("Precision, Recall, and F1", "model evaluation",
     "Precision P = TP / (TP + FP) measures what fraction of positive predictions are correct. "
     "Recall R = TP / (TP + FN) measures what fraction of actual positives are detected. "
     "The F1 score F1 = 2PR/(P+R) is their harmonic mean, balancing the two. "
     "ROC curves plot true positive rate against false positive rate across thresholds; "
     "the area under the curve (AUC) summarises classifier performance independent of threshold. "
     "For imbalanced classes, average precision (area under the precision-recall curve) is often more informative."),

    ("Markov Chains", "probability",
     "A Markov chain is a stochastic process where the probability of each state depends only on the previous state: "
     "P(X_{n+1}=j | X_n=i, X_{n-1},...) = P(X_{n+1}=j | X_n=i). "
     "The transition matrix T where T_{ij} = P(X_{n+1}=j | X_n=i) fully characterises a finite Markov chain. "
     "An ergodic chain has a unique stationary distribution π satisfying πT=π, which the chain converges to over time. "
     "Markov Chain Monte Carlo (MCMC) methods use carefully constructed chains to sample from complex target distributions."),

    ("Backpropagation", "deep learning",
     "Backpropagation computes the gradient of the loss with respect to each parameter by applying the chain rule "
     "through the computation graph. For a layer with input x, weight W, and output y=Wx, "
     "the gradient ∂L/∂W = (∂L/∂y)xᵀ and ∂L/∂x = Wᵀ(∂L/∂y). "
     "Automatic differentiation frameworks (PyTorch, JAX) implement this efficiently by tracking operations "
     "in a computation graph and reversing through it during the backward pass. "
     "Vanishing gradients in deep networks are mitigated by ReLU activations, residual connections, and careful initialisation."),

    ("Clustering: K-Means", "unsupervised learning",
     "K-means partitions n observations into k clusters by alternating between two steps: "
     "(1) assign each point to the nearest centroid, (2) recompute centroids as the mean of assigned points. "
     "The algorithm minimises the within-cluster sum of squared distances (inertia). "
     "Convergence is guaranteed but to a local minimum; multiple random initialisations (k-means++) reduce sensitivity. "
     "The elbow method and silhouette coefficient help select the number of clusters k."),

    ("Hypothesis Testing", "statistics",
     "Hypothesis testing evaluates whether observed data provide sufficient evidence against a null hypothesis H₀. "
     "The p-value is the probability of observing results at least as extreme as the data, assuming H₀ is true. "
     "If p < α (typically 0.05), H₀ is rejected at significance level α. "
     "Type I error (false positive) occurs when H₀ is true but rejected; its rate is α. "
     "Type II error (false negative) occurs when H₀ is false but not rejected; its complement is the power of the test. "
     "Multiple testing corrections (Bonferroni, Benjamini-Hochberg) control error rates across families of tests."),

    ("Embedding and Vector Spaces", "deep learning",
     "Word embeddings represent discrete tokens as dense real-valued vectors in a shared semantic space. "
     "Word2Vec learns embeddings by predicting context words from a target (skip-gram) or predicting a target from context words (CBOW). "
     "Semantically similar words cluster near each other: the cosine similarity cos(u,v) = (u·v)/(‖u‖‖v‖) "
     "measures the angle between vectors irrespective of magnitude. "
     "Transformer models produce contextualised embeddings that depend on the entire input sequence, "
     "capturing polysemy and long-range dependencies that static embeddings cannot represent."),
]


def _expand_concept(rng: random.Random, name: str, domain: str, body: str) -> str:
    """Wrap a concept with a heading and a generated paragraph of additional context."""
    elaborations = [
        f"Understanding {name} is fundamental to work in {domain}.",
        f"{name} appears frequently in both theoretical and applied {domain}.",
        f"Practitioners in {domain} rely on {name} as a foundational tool.",
        f"A solid grasp of {name} enables clearer reasoning across {domain}.",
        f"Research in {domain} has refined the theory of {name} considerably over the past decades.",
    ]
    caveats = [
        "However, the assumptions underlying this approach should be verified for each application.",
        "In practice, implementation details and numerical stability are important considerations.",
        "Empirical results sometimes diverge from theoretical predictions, requiring careful validation.",
        "The choice of hyperparameters can significantly affect results and warrants systematic tuning.",
        "Extensions and generalisations of this concept remain active areas of research.",
    ]
    lines = [
        f"## {name}",
        f"",
        f"*Domain: {domain}*",
        f"",
        body,
        f"",
        rng.choice(elaborations) + " " + rng.choice(caveats),
    ]
    return "\n".join(lines)


def generate_datascience(rng: random.Random, count: int) -> str:
    """Generate *count* data-science concept explanations."""
    sections = []
    pool = list(_DS_CONCEPTS)
    for i in range(count):
        name, domain, body = pool[i % len(pool)]
        # Shuffle sentence order slightly on repeat passes to create variation
        if i >= len(pool):
            sentences = body.split(". ")
            rng.shuffle(sentences)
            body = ". ".join(sentences)
        sections.append(_expand_concept(rng, name, domain, body))
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic lore and data-science text for the Grimoire corpus"
    )
    parser.add_argument(
        "--output", default="data/corpus/saga/",
        help="Output directory (default: data/corpus/saga/)",
    )
    parser.add_argument(
        "--group", choices=["lore", "datascience", "all"], default="all",
        help="Which content group to generate (default: all)",
    )
    parser.add_argument(
        "--count", type=int, default=2000,
        help="Number of entries to generate per group (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    groups = ["lore", "datascience"] if args.group == "all" else [args.group]

    for group in groups:
        dest = out / f"synth_{group}.txt"
        if dest.exists() and not args.force:
            print(f"  [skip] {dest} already exists (pass --force to overwrite)")
            continue

        rng = _rng(args.seed)
        print(f"  Generating {args.count} {group} entries...")
        if group == "lore":
            text = generate_lore(rng, args.count)
        else:
            text = generate_datascience(rng, args.count)

        dest.write_text(text, encoding="utf-8")
        size_kb = dest.stat().st_size // 1024
        print(f"  ✔ {dest} ({size_kb} KB)")

    print("\nDone. Run the Preprocess tab to tokenize the new files.")


if __name__ == "__main__":
    main()
