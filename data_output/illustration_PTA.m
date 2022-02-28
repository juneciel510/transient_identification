pi=200; % initial pressure, bar
k=100;  % permeability, mD
h=10;    % thickness, m
poro=0.3; % porosity
rw=0.1; % well radius, m
q=100;  % m3/day
mu=1; % cP
ct=4*10^-5; % bar^-1
B=1;

t=0.02:.02:100; % hr
l=length(t);
% tD=0.0003553*k*t/(poro*mu*ct*rw^2); % 1.8.3
tD=0.0003553*k/(poro*mu*ct*rw^2) * t; % dimensionless time

pwD=-0.5*expint(1./(4*tD)); 

% a = - 0.25 / (0.0003553*k/(poro*mu*ct*rw^2));
% b = -0.5*18.66*q*B*mu/(k*h);

p=pi-18.66*q*B*mu/(k*h)*pwD;
% p=18.66*q*B*mu/(k*h)*pwD;
pe=p+0.1*randn(1,l);

figure;
plot(t,pe,'.r'); grid on; hold on;
plot(t,p,'-k'); grid on; hold on;
legend('noisy', 'noiseless')
xlabel('Time, hr','FontSize',12);
ylabel('Pressure, bar','FontSize',12);

figure;
semilogx(t,pe,'.r'); grid on; hold on;
semilogx(t,p,'-k'); grid on; hold on;
legend('noisy', 'noiseless')
xlabel('Time, hr','FontSize',12);
ylabel('Pressure, bar','FontSize',12);