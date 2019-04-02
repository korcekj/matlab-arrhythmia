clear
load dataarytmiasrdca

datainnet=NDATA'; % vstupne data
dataoutnet=parseClassify(typ_ochorenia); % vystupne data

%  vytvorenie struktury siete
pocet_neuronov=30;  % definujte pocet neuronov v skrytej vstve
net = patternnet(pocet_neuronov); % NS

skupina1 = (1:88);
skupina2 = (89:176);
skupina3 = (177:264);
skupina4 = (265:352);
skupina5 = (353:442);

% net.divideFcn='dividerand';
% net.divideParam.trainRatio=0.6;
% net.divideParam.valRatio=0.2;
% net.divideParam.testRatio=0.2;

net.divideFcn = 'divideind';

% Indexove rozdelenie skupin 1,3,5; 4; 2;
net.divideParam.trainInd = [skupina1, skupina3, skupina5];
net.divideParam.valInd = skupina4; 
net.divideParam.testInd = skupina2;

% % Indexove rozdelenie skupin 1,2,3; 4; 5;
% net.divideParam.trainInd = [skupina1, skupina2, skupina3];
% net.divideParam.valInd = skupina4; 
% net.divideParam.testInd = skupina5;

% % Indexove rozdelenie skupin 3,4,5; 1; 2;
% net.divideParam.trainInd = [skupina3, skupina4, skupina5];
% net.divideParam.valInd = skupina1; 
% net.divideParam.testInd = skupina2;

% % Indexove rozdelenie skupin 1,2,4; 3; 5;
% net.divideParam.trainInd = [skupina1, skupina2, skupina5];
% net.divideParam.valInd = skupina3; 
% net.divideParam.testInd = skupina4;

% % Indexove rozdelenie skupin 2,4,5; 1; 3;
% net.divideParam.trainInd = [skupina1, skupina2, skupina5];
% net.divideParam.valInd = skupina3; 
% net.divideParam.testInd = skupina4;

% nastavenie parametrov trenovania
net.trainParam.goal = 1e-6;       % ukoncovacia podmienka na chybu.
net.trainParam.show = 10;           % frekvencia zobrazovania chyby
net.trainParam.epochs = 250;    % maximalny pocet trenovacich epoch.
net.trainParam.max_fail = 10;

% trenovanie neuronovej siete
net = train(net,datainnet,dataoutnet);

% struktura NS
% view(net);

% testovanie NS
outnetsim = sim(net, datainnet);

% chyba NS a dat
err=(outnetsim-dataoutnet);

% Zastupenie po skupinach
% counter(typ_ochorenia)'

% plot confusion
y=net(datainnet);
plotconfusion(dataoutnet, y);