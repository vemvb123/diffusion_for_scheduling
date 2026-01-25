


Kan encode maskin slik:



Skal outputte actions....
Outputter slik:
Adj matrise



Eller actions... men vet ikke helt...

Burde fungere slikA



Hvordan bruke:
Outputter action, som er map assignment


Den outputter en action, som er en maop assignment



Inputter:
Feature vector, for hver op og ma
Gjer ved at inputte en liste


kan ogsa bruke adj, der hver celle i adj er som en feature vector.
Eller kan kanskje ogsa bare lage en feature vector, som har forskjellig info,
eks:
[prosesseringstid, opid, jobid, maid]
Eller for hver av disse lager jeg en feature vector,
sa man har flere feature vectorer
Sa nar man seinere inputter en viss prosseseringstid, far man tilbake en ciss feature vector




Henter det ut slik:
Inputer matriser der data ligger langs en rad, slik som i td.
Putter inn forskjellige matriser over forskjellige input channels.
Inputter inn i vanlig unet. Da vil en feature vector ta hoyde for sig selv, og naboene.

Bruker unet fordi:
Meninga er at det skal ga raskt, det var hele poenget med diffusion
Det er lett a komme i gang med for nu.

Input shapes blir da:
b,1,w,h for en matrise 
Jeg putter en matrise over hver kanal, som da gir:
b, feature typer,h,w

output channels er til unsket mengde feature dim.
b, mengde features, h, w

Sa na for hver operasjon, sa har jeg en eller flere f vectors, gitt av hw kombinasjonen
Henter ut feature vectoren for hver operasjon.
Sa det da blir for en op: b, feature dim
og for hele datastrukturen: b, mengde features, feature dim

sa bruker 1d unet, som gar over en feature vector 1 og 1,
Hvis det er mulig a distinksere feature typer, sa legger jeg forskjellige vectorer langs 
hver input channel, hvis ikke ligger de alle langs samme channel, som en feature


# TODO men.. jeg kan vell ikke velge hoyrde og w av output, eller?
Sa kan jeg ha noe som target: en celle for hver mulig allokasjon,
som er ma, op: sannsynlighet for allokasjon (1 eller 0)

