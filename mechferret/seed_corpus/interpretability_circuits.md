# Mechanistic interpretability: known circuits and methods

Mechanistic interpretability reverse-engineers the algorithms learned by neural
networks into human-understandable components: circuits of attention heads and
MLP neurons that implement specific behaviours.

## Indirect Object Identification (IOI)

The IOI task asks a model to complete sentences like "When John and Mary went to
the store, Mary gave a drink to" with the indirect object ("John"). Wang et al.
(2022) identified a circuit in a small text transformer involving several head classes:
duplicate-token heads detect the repeated name, S-inhibition heads suppress the
subject, and name-mover heads copy the correct name to the final position. A
small number of negative name-mover heads push in the opposite direction. The
circuit is validated by path patching and by ablation: ablating name-mover
heads sharply reduces the logit difference between the correct and incorrect
name.

## Induction heads

Induction heads (Olsson et al. 2022) implement in-context copying: given a
sequence "[A][B] ... [A]", they attend from the second [A] to the token after
the first [A] and predict [B]. They are detected by attention-pattern analysis
(the previous-token head feeds a later induction head) and by ablation on
repeated-sequence tasks. Induction heads are associated with the emergence of
in-context learning during training.

## Logit lens

The logit lens (nostalgebraist 2020) projects intermediate residual-stream
activations through the unembedding to read the model's "current guess" at each
layer. It reveals the layer at which a prediction crystallises and is a cheap
first-pass localisation method before causal patching.

## Activation patching and causal tracing

Activation patching (a.k.a. causal tracing in ROME, Meng et al. 2022) copies a
clean activation into a corrupted run and measures how much of the correct
behaviour is restored. It localises *where* information is causally used.
Factual associations are stored predominantly in mid-layer MLPs at the last
subject token.

## Rigor

Trustworthy mechanistic claims require negative controls (a site that should not
matter), reproducibility across seeds and prompts, and triangulation: an
ablation effect should agree with direct logit attribution and with the head's
attention pattern before a head is named a "name mover" or "induction head".
