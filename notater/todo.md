
tips:
end kjøringa når loss er for det meste under 0.015-0.010
ops... for mk01 ser det ut som om loss gikk til rundt 0.01-0.02 for test (litt lavere for train, men ikke veldig).. den trente i 15 epoker
...
kjøringer:
trene mk06
trene mk07
target modell mk08


..
videre:
batch replacement mk01
batch replacement mk10
noe som kort viser mautil for trent modell der
noe kort statistisk som viser machine busy tid = høyere makespan mk01
trening mk07/08/09



skjekk om lessutil modell faktisk skedulerer mindre
fikse sched problem mk05
skaff konf intervall for mk05 og mk03
vent med mk03 til du er sikker du veit maskestørrelsen
skaff resultater for strategier
..
stoppet å trene mk03 modell fordi jeg trur den er ferdig, men må skjekke prestasjon.
Hvis ikke, så fortsett trening

trene target modell mk06
..

Fikse infeas..
Ide:
Rapporter infeas for hvert steg i fiksinga, slik at du kan se når det går gærent,
sammenlign stega med mk01
.. trur bare kan fikse på main branchen, main og ikke-main har vell det samme problemet.

...

OPS: 
mk03?? kan ha dårlig prest i infeasibility fordi jeg bruker feil maske.
Brukte 60 maske på mk01, og fikk da mye mer feil enn når jeg brukte maske for 64 (det modellen var trent på)
må prøve forskjellige maskeverdier for mk03









X kjore for mk02, se nar stopper, om stopper med en gang pa epoke 2
X kjore for mk01, se om stopper i det hele tatt
...
Den stoppet a lare i midten av mk01, for mk02 stoppet den i i helt av starten av 2 epoke. men kan hende ogsa var noe tidligere
Dette betyr at det ma vera noe med koden


.... hvis modell for mk01 ikke feiler, sa er det ikke er koden i seg selv. det er datasettet eller modellen i seg selv, eller er sa det noe med koden i seg selv
bruk eksisterende kode til a lage enda et mk01 datasett
tren modell igjen for det nye lagde datasettet for mk01
..... hvis modell ikke klarer det, veit jeg det er noe med datasettet, hvis modellen klarer det er det noe med modellen i seg selv



....
for a kjore, ma nok...

X endre sa blir sammenpressa i den ene funksjonen.
X legge a masker
endre filnavn pa grafer og slikt
passe a overskriver forgje confidence fila, ikke bare legger til mer
se at fikser sched feil

tips... på stor benchmark kan du kanskje bruke batch erstattninga






Skeduleringsfeil kommer av fogjande:
Nar det er mange oppgaver, blir det stort spenn av aktuelle verdier.

Med 55 verdier mellom 0 og 1, sa er den forte verdien nesten a 0.02
Nar det ikke er feil, sa ser det ut som den minste verdien er hoyere enn det ville trodd, eks 0.04

Jeg tror hvis man har storre spenn, sa kan det kanskje lose problemet. Na er det mellom 0 og 1.
Men kan hende diff modellen eks tillater mellom -2 og 2


Vrdn best fikse skeduleringsfeil, nar dem faktisk hender


Sett:
Gjerne den fyste, altsa minste oppgaven fra en jobb, der det ikke er assignment
Der det er to verdier assignet, kan begge verdiene vere hoye, men de er gjerne begge lavere enn neste verdi

Jeg tror nar en er skedulert fur en pred, sa er de begge veldig like i verdi

Det kan hende sked feil fordi noen fa verdier er hoye, men utenfor gyldige steder
....

nar mangler verdi:
flytt storste verdi til gyldig pos .. ser ut til a alltid vere en fyrste rad som mangler verdi

nar pred feil:
swap fyr og etter verdiene, de er ganske like i verdi uansett, som betyr at de ikke skeduleres langt unna hverandre, kanskje rett etter hverandre














Test guiding for inference
test guiding for trening

...

Noe a fikse..
det er faktisk noe feil med skeduleringa, det kan ses at operasjoner venter unødvendig lenge med å bli skedulert.
Det er ikke noe i skedualet, men noe i den faktisk skeduleringsalgoritma

Runding:
Når runder så tar de globalt største verdiene, og setter dem for allokasjoner,
så resulterer det gjerne i at skedual ikke har allokasjoner for noen operasjoner




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





