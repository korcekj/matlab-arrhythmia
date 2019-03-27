clear
load dataarytmiasrdca

datainnet=NDATA';
dataoutnet=parseClassify(typ_ochorenia);

%  vytvorenie struktury siete
pocet_neuronov=10;  % definujte pocet neuronov v skrytej vstve
net = patternnet(pocet_neuronov);

net.divideFcn='dividerand';
net.divideParam.trainRatio=0.6;
net.divideParam.valRatio=0.2;
net.divideParam.testRatio=0.2;

% nastavenie parametrov trenovania
% definujte parametre trenovania siete
net.trainParam.goal = 1e-8;       % ukoncovacia podmienka na chybu.
net.trainParam.show = 10;           % frekvencia zobrazovania chyby
net.trainParam.epochs = 1000;        % maximalny pocet trenovacich epoch.
net.trainParam.max_fail = 100;

% trenovanie neuronovej siete
net = train(net,datainnet,dataoutnet);

% simulacia vystupu NS pre trenovacie data
% testovanie NS
outnetsim = sim(net,datainnet);

% chyba NS a dat
err=(outnetsim-dataoutnet);

% plot confusion
y=net(datainnet);
plotconfusion(dataoutnet, y);