
# TODO

Når skal skjekke chache inference... Skjekk at vediarnir or n_ops vert riktige.

lage funk for se om order respekterer seq
lage så får masse løsninger, så plukker den beste

bytt ut diffusion scheduerling med sånn annen type.. ikke schedule som i fjssp, men sånn schedule som generelt i diffusion

# tenke over
OPS:
når får fra inference...
burde nok gjøre noe skjekk på om ikke op skeduleres til ugyldig ma
Kan gjøre dette med å bare skjekke den ene adj greia som viser hvilken op kan være på hvilken ma.

Lag noe som kan få resultater fra en batch i inference.
Så jeg eks får et snitt for:
feasibility - genereres nok op, gis op til riktige ma, i hvor stor grad op følger sekvens
makespan
generasjonstid (dette blir ikke et snitt, med for hele batchen)


# gjøre
før trening... Ta flere av instansene inn i et testfolder. Du kan ikke trene på instanser, så bruke de samme når du skjekker inference
lage 444 datasett på nytt
trene 444 modell
lag inferene tingen for order også..


# raskere inference
## llm
block wise kv caching .... tror kanskje noe av dette funker som maske .. man kan gjøre noe for at modell ignore et sted, ikke sett
confidence-aware parallell decoding - tokens confidently predicted gets unmasked in parallell
kanskje man kan ha noe confidence for hele sched, så bruke ddim til a stoppe tidlig.

Kan rpøve å implementere block wise kv caching og condfidence aware parrallell decoding, kan teste med 444
Men burde kanskje endre treninga også etterhvert, eller?

## andre metoder
ddim
progressive / learned distillation
latent space diffuion - opererer i compressed space
feture caching






# nøkkler tensordict

start_op_per_job
end_op_per_job
proc_times
pad_mask
ops_adj
job_ops_adj
ops_job_map
ops_sequence_order
start_times
finish_times
ma_assignment
busy_until
num_eligible
next_op
ops_ma_adj
op_scheduled
job_in_process
reward
time
job_done
done
action_mask
lbs
is_ready
terminated
proc_times




# Kiler

tspdiffuserTSPDiffuser: Diffusion Models as Learned Samplers for
Traveling Salesperson Path Planning Problems
https://arxiv.org/pdf/2406.02858

difformer
https://arxiv.org/pdf/2301.09474

diffusion moas plug and priors
https://arxiv.org/pdf/2206.09012

Planning with Diffusion for Flexible Behavior Synthesis
https://arxiv.org/pdf/2205.09991

Exploring the Boundary of Diffusion-based Methods
for Solving Constrained Optimization
https://arxiv.org/pdf/2502.10330

DIFUSCO: Graph-based Diffusion Solvers
for Combinatorial Optimization
https://arxiv.org/pdf/2302.08224





