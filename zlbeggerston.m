clear
close all

% Parameters

sigma  = 1/1.16;         % Elasticity if intertemporal substitution.
beta   = 0.997;          % Discount factor.
alpha  = 0.7747;         % Price stickiness (measure of firms that cannot reset their prices in a given period). 
theta  = 12.7721;        % Elasticity of substitution.
omega  = 1.5692;         % Frisch elasticity of labour supply.
phi_pi = 1.5;           % Coefficient on pi in Taylor rule.
phi_y  = 0.5/4;          % Coefficient on y in Taylor rule.
rshock = -0.0104;           % Shock to preference or productivity that brings the economy to a liquidity trap.
mu     = 0.9030;            % Probability of remaining in a liquidity trap.
gstim  = [0,0.05];       % possible government expenditures at zero lower bound.

% Reduced form parameters

kappa = (1-alpha)*(1-alpha*beta)/alpha*(1/sigma+omega)/(1+omega*theta);

psi = 1/(1/sigma+omega);

% Theoretical multiplier according to equation (30) in Eggertson
% corresponding to the trap case
dydg = ((1-mu)*(1-beta*mu)-mu*kappa*psi)/((1-mu)*(1-beta*mu)-sigma*mu*kappa);

% Transition matrix.

prob = 1; % perceived probability of staying in the no-trap regime 
          % if this probability equals 1 then moving to the trap regime is an MIT shock
          
T    = [prob,1-prob;1-mu,mu];
% at mu = 0.9030 the system is only stable at values of prob close to 1. For
% example, prob = 0.99 is already too big. However, prob = 0.99 does work if
% mu=0.85.

% Steady state values.

y_ss = 0;
pi_ss = 0;
i_ss = -log(beta);
r_ss = i_ss;
g_ss = 0;

% Set up the two systems symbolically for first government expenditure
% value

e_n = zeros(5,2);
e_t = zeros(5,2);
f_n = zeros(5,5,2);
f_t = zeros(5,5,2);

for iloop = 1:2
    
gshock = gstim(iloop);

syms ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp
%ym : lagged value (so current value of state variable)
%y  : current-period outcome
%ypn: next-period's value in the 'no-trap' state
%ypt: next-period's value in the 'trap' state
%y  : output
%pi : inflation (but current inflation is ppp not pi to avoid confusion)
%g  : government expenditure
%i  : nominal interest rate
%r  : real interest rate

no_trap = [-y+T(1,1)*ypn+T(1,2)*ypt-sigma*(i-(T(1,1)*pipn+T(1,2)*pipt)-r)+g-(T(1,1)*gpn+T(1,2)*gpt);
    -ppp+kappa*y+kappa*psi*(-1/sigma*g)+beta*(T(1,1)*pipn+T(1,2)*pipt);
    -i+r+phi_pi*ppp+phi_y*y;
    -r+r_ss;
    -g+g_ss;];

trap = [-y+T(2,1)*ypn+T(2,2)*ypt-sigma*(i-(T(2,1)*pipn+T(2,2)*pipt)-r)+g-(T(2,1)*gpn+T(2,2)*gpt);
    -ppp+kappa*y+kappa*psi*(-1/sigma*g)+beta*(T(2,1)*pipn+T(2,2)*pipt);
    -i+0;
    -r+r_ss+rshock;
    -g+gshock;];

% Note that the maginitude of the shock rshock does not really affect the 
% fiscal muliplier due to linearization
% However, it does affect the IRF

% If you uncomment the below code you will instead find the multiplier when
% the economy is not in a liquidity trap.

% trap = [-y+T(2,1)*ypn+T(2,2)*ypt-sigma*(i-(T(2,1)*pipn+T(2,2)*pipt)-r)+g-(T(2,1)*gpn+T(2,2)*gpt);
%     -ppp+kappa*y+kappa*psi*(-1/sigma*g)+beta*(T(2,1)*pipn+T(2,2)*pipt);
%     -i+rm+phi_pi*ppp+phi_y*y;
%     -r+r_ss+rshock;
%     -g+gshock;];

% Get the matrices. 
% Note that derivative is taking with respect to output, inflation,
% government expenditure, nominal interest rate, and targeted nominal
% interest rate
A_n = jacobian(no_trap,[ym pim gm im rm]); A_n = subs(A_n,[ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp],[y_ss y_ss y_ss y_ss pi_ss pi_ss pi_ss pi_ss g_ss g_ss g_ss g_ss i_ss i_ss i_ss r_ss r_ss r_ss]);
%Here we pin down the order of the variables, that is,
%output is the first variable,
%inflation is the second, etc.
%This order is important to understand E_n,F_n,E_t, and F_t


B_n = jacobian(no_trap,[y ppp g i r]); B_n = subs(B_n,[ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp],[y_ss y_ss y_ss y_ss pi_ss pi_ss pi_ss pi_ss g_ss g_ss g_ss g_ss i_ss i_ss i_ss r_ss r_ss r_ss]);

C_nn = jacobian(no_trap,[ypn pipn gpn ip rp]); C_nn = subs(C_nn,[ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp],[y_ss y_ss y_ss y_ss pi_ss pi_ss pi_ss pi_ss g_ss g_ss g_ss g_ss i_ss i_ss i_ss r_ss r_ss r_ss]);

C_nt = jacobian(no_trap,[ypt pipt gpt ip rp]); C_nt = subs(C_nt,[ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp],[y_ss y_ss y_ss y_ss pi_ss pi_ss pi_ss pi_ss g_ss g_ss g_ss g_ss i_ss i_ss i_ss r_ss r_ss r_ss]);

D_n = subs(no_trap,[ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp],[y_ss y_ss y_ss y_ss pi_ss pi_ss pi_ss pi_ss g_ss g_ss g_ss g_ss i_ss i_ss i_ss r_ss r_ss r_ss]);

A_t = jacobian(trap,[ym pim gm im rm]); A_t = subs(A_t,[ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp],[y_ss y_ss y_ss y_ss pi_ss pi_ss pi_ss pi_ss g_ss g_ss g_ss g_ss i_ss i_ss i_ss r_ss r_ss r_ss]);

B_t = jacobian(trap,[y ppp g i r]); B_t = subs(B_t,[ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp],[y_ss y_ss y_ss y_ss pi_ss pi_ss pi_ss pi_ss g_ss g_ss g_ss g_ss i_ss i_ss i_ss r_ss r_ss r_ss]);

C_tn = jacobian(trap,[ypn pipn gpn ip rp]); C_tn = subs(C_tn,[ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp],[y_ss y_ss y_ss y_ss pi_ss pi_ss pi_ss pi_ss g_ss g_ss g_ss g_ss i_ss i_ss i_ss r_ss r_ss r_ss]);

C_tt = jacobian(trap,[ypt pipt gpt ip rp]); C_tt = subs(C_tt,[ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp],[y_ss y_ss y_ss y_ss pi_ss pi_ss pi_ss pi_ss g_ss g_ss g_ss g_ss i_ss i_ss i_ss r_ss r_ss r_ss]);

D_t = subs(trap,[ym y ypt ypn pim ppp pipt pipn gm g gpt gpn im i ip rm r rp],[y_ss y_ss y_ss y_ss pi_ss pi_ss pi_ss pi_ss g_ss g_ss g_ss g_ss i_ss i_ss i_ss r_ss r_ss r_ss]);

A_n = double(A_n);

B_n = double(B_n);

C_nn = double(C_nn);

C_nt = double(C_nt);

A_t = double(A_t);

B_t = double(B_t);

C_tn = double(C_tn);

C_tt = double(C_tt);

D_n = double(D_n);

D_t = double(D_t);

%% Initial guesses.

E_n = zeros(5,1);
E_t = zeros(5,1);

F_n = zeros(5,5);
F_t = zeros(5,5);

metric = 1;

% Solve the problem.

while metric>1e-13
    
    % First calculate new guesses given the previous:

    E_n_new = (B_n+C_nn*F_n+C_nt*F_t)\(-(C_nn*E_n+C_nt*E_t+D_n));
    E_t_new = (B_t+C_tn*F_n+C_tt*F_t)\(-(C_tn*E_n+C_tt*E_t+D_t));

    F_n_new = (B_n+C_nn*F_n+C_nt*F_t)\(-A_n);
    F_t_new = (B_t+C_tn*F_n+C_tt*F_t)\(-A_t);
    
    % Then update the guesses for the next iteration:

    E_n = E_n_new;
    E_t = E_t_new;
    F_n = F_n_new;
    F_t = F_t_new;
    
    % Check if the Euler equation is close to zero (at state=ones(5,1)):    
    metric = max(max(abs([A_n*ones(5,1)+(B_n+C_nn*F_n+C_nt*F_t)*(E_n+F_n*ones(5,1))+(C_nn*E_n+C_nt*E_t+D_n) A_t*ones(5,1)+(B_t+C_tn*F_n+C_tt*F_t)*(E_t+F_t*ones(5,1))+(C_tn*E_n+C_tt*E_t+D_t)])));
    disp(metric)
end

% Done problem solved

    e_n(:,iloop) = E_n;
    e_t(:,iloop) = E_t;
    f_n(:,:,iloop) = F_n;
    f_t(:,:,iloop) = F_t;

% Set up the two systems symbolically for first government expenditure
% value

end


%% Set the length of the impulse response.

T_imp = 50;

% Calculate paths conditional on particular state

X10 = zeros(5,T_imp,2);
X20 = X10;
X30 = X10;
X40 = X10;

% Ok done. Let's calculate the impulse responses when the negative shock 
% lasts for 10, 20, 30, and 40 periods

for iloop = 1:2

E_t=e_t(:,iloop); 
E_n=e_n(:,iloop); 
F_t=f_t(:,:,iloop);
F_n=f_n(:,:,iloop);

X10(:,2,iloop) = E_t+F_t*X10(:,1,iloop);


for i = 2:T_imp
    
    if i<10
        X10(:,i+1,iloop) = E_t+F_t*X10(:,i,iloop);
    else
        X10(:,i+1,iloop) = E_n+F_n*X10(:,i,iloop);
    end
    if i<20
        X20(:,i+1,iloop) = E_t+F_t*X20(:,i,iloop);
    else
        X20(:,i+1,iloop) = E_n+F_n*X20(:,i,iloop);
    end
    if i<30
        X30(:,i+1,iloop) = E_t+F_t*X30(:,i,iloop);
    else
        X30(:,i+1,iloop) = E_n+F_n*X30(:,i,iloop);
    end
    if i<40
        X40(:,i+1,iloop) = E_t+F_t*X40(:,i,iloop);
    else
        X40(:,i+1,iloop) = E_n+F_n*X40(:,i,iloop);
    end
    
end

end

figure(1)
subplot(2,3,1)
plot(100*(X10(1,:,2)),'-*','LineWidth',2.0,'color','b');
hold on
plot(100*(X20(1,:,2)),'LineWidth',1.0,'color','b');
plot(100*(X30(1,:,2)),'LineWidth',1.0,'color','b');
plot(100*(X40(1,:,2)),'LineWidth',1.0,'color','b');
plot(100*(X10(1,:,1)),'-*','LineWidth',2.0,'color','k');
plot(100*(X20(1,:,1)),'LineWidth',1.0,'color','k');
plot(100*(X30(1,:,1)),'LineWidth',1.0,'color','k');
plot(100*(X40(1,:,1)),'LineWidth',1.0,'color','k');
title('Output, $Y_t$','Interpreter','latex')

iloop = 2;
subplot(2,3,2)
plot(100*(X10(2,:,2)),'-*','LineWidth',2.0,'color','b');
hold on
plot(100*(X20(2,:,2)),'LineWidth',1.0,'color','b');
plot(100*(X30(2,:,2)),'LineWidth',1.0,'color','b');
plot(100*(X40(2,:,2)),'LineWidth',1.0,'color','b');
plot(100*(X10(2,:,1)),'-*','LineWidth',2.0,'color','k');
plot(100*(X20(2,:,1)),'LineWidth',1.0,'color','k');
plot(100*(X30(2,:,1)),'LineWidth',1.0,'color','k');
plot(100*(X40(2,:,1)),'LineWidth',1.0,'color','k');
title('Inflation, $\pi_t$','Interpreter','latex')

subplot(2,3,3)
plot(100*(X10(3,:,2)),'-*','LineWidth',2.0,'color','b');
hold on
plot(100*(X20(3,:,2)),'LineWidth',1.0,'color','b');
plot(100*(X30(3,:,2)),'LineWidth',1.0,'color','b');
plot(100*(X40(3,:,2)),'LineWidth',1.0,'color','b');
plot(100*(X10(3,:,1)),'-*','LineWidth',2.0,'color','k');
plot(100*(X20(3,:,1)),'LineWidth',1.0,'color','k');
plot(100*(X30(3,:,1)),'LineWidth',1.0,'color','k');
plot(100*(X40(3,:,1)),'LineWidth',1.0,'color','k');
title('Government Spending, $G_t$','Interpreter','latex')

subplot(2,3,4)
plot(100*(X10(4,:,2)+i_ss),'-*','LineWidth',2.0,'color','b');
hold on
plot(100*(X20(4,:,2)+i_ss),'LineWidth',1.0,'color','b');
plot(100*(X30(4,:,2)+i_ss),'LineWidth',1.0,'color','b');
plot(100*(X40(4,:,2)+i_ss),'LineWidth',1.0,'color','b');
title('Nominal Interest Rate, $i_t$','Interpreter','latex')
% To ease interpretation, steady state value is added (since the u_t variables 
% are all relative to no-trap steady state values
% and thus, zero in the no-trap regime)

subplot(2,3,5)
plot(100*(X10(5,:,2)+r_ss),'-*','LineWidth',2.0,'color','b');
hold on
plot(100*(X20(5,:,2)+r_ss),'LineWidth',1.0,'color','b');
plot(100*(X30(5,:,2)+r_ss),'LineWidth',1.0,'color','b');
plot(100*(X40(5,:,2)+r_ss),'LineWidth',1.0,'color','b');
title('Real Interest Rate, $r_t$','Interpreter','latex')
% To ease interpretation, steady state value is added (since the u_t variables 
% are all relative to no-trap steady state values
% and thus, zero in the no-trap regime)

xSize = 20; ySize = 16;
set(gcf,'Units','centimeters','Position',[0 0 xSize ySize],'PaperUnits','centimeters' ...
     ,'PaperPosition',[0 0 xSize ySize],'PaperSize',[xSize-2 ySize-1],'PaperPositionMode','auto')

print -dpdf output1.pdf


%Calculate expected paths, that is, the IRFs

v   = zeros(T_imp,2);
irf = ones(5,T_imp,2);

v(1,:) = [1,0];
v(2,:) = [0,1];

for i = 2:T_imp
    v(i+1,:) = v(i,:)*T;
end

for iloop = 1:2
    
% Initial value:

uss = zeros(5,1);
ussg = uss;

Ex_n  = uss;
Ex_t  = uss;
Exp   = uss;

% in period 1 I am in state 1 and in steady state (uss)
% shock occurs in period 2 and I know with certainty that I am in state 2,
% so expected value is equal to actual outcome

E_t  = e_t(:,iloop);
F_t  = f_t(:,:,iloop);

Ex_t(:,2) = E_t + F_t*uss;
Ex_n(:,2) = E_n + F_n*uss; 
Exp(:,2)  = E_t + F_t*uss;  
% value for Ex_n(:,2) is irrelevant (state has zero  probability since I know with certainty that I am in the trap in t=2)
% Exp(:,2) = Ex_t(:,2) since I know with certainty I am in trap state in t=2

for i = 2:T_imp-1
    
    Ex_t(:,i+1) = T(2,2)*(v(i,2)/v(i+1,2))*(E_t+F_t*Ex_t(:,i))+T(1,2)*(v(i,1)/v(i+1,2))*(E_t+F_t*Ex_n(:,i));
    Ex_n(:,i+1) = T(2,1)*(v(i,2)/v(i+1,1))*(E_n+F_n*Ex_t(:,i))+T(1,1)*(v(i,1)/v(i+1,1))*(E_n+F_n*Ex_n(:,i));
    
    % Calculate the IRF. It is the IRF for the
    % expectation of future variables
    Exp(:,i+1) = v(i+1,1)*Ex_n(:,i+1)+v(i+1,2)*Ex_t(:,i+1);
    
end

irf(:,:,iloop) = Exp;

end

%Both figures plot the expected path with and without government expenditure shock
%In figure(1), the conditional IRFs are for the case without g shock
%In figure(2), the conditional IRFs are for the case with g shock

figure(2)
subplot(2,3,1)
plot(100*(irf(1,:,1)),'LineWidth',2,'color','k');
hold on
plot(100*(irf(1,:,2)),'LineWidth',2,'color','b');
plot(100*(X10(1,:,iloop)),'-*','LineWidth',2.0,'color','b');
plot(100*(X20(1,:,iloop)),'LineWidth',1.0,'color','b');
plot(100*(X30(1,:,iloop)),'LineWidth',1.0,'color','b');
plot(100*(X40(1,:,iloop)),'LineWidth',1.0,'color','b');
title('Output, $Y_t$','Interpreter','latex')

subplot(2,3,2)
plot(100*(irf(2,:,1)),'LineWidth',2,'color','k');
hold on
plot(100*(irf(2,:,2)),'LineWidth',2,'color','b');
plot(100*(X10(2,:,iloop)),'-*','LineWidth',2.0,'color','b');
plot(100*(X20(2,:,iloop)),'LineWidth',1.0,'color','b');
plot(100*(X30(2,:,iloop)),'LineWidth',1.0,'color','b');
plot(100*(X40(2,:,iloop)),'LineWidth',1.0,'color','b');
title('Inflation, $\pi_t$','Interpreter','latex')

subplot(2,3,3)
plot(100*(irf(3,:,2)),'LineWidth',2,'color','b');
hold on
plot(100*(X10(3,:,iloop)),'-*','LineWidth',2,'color','b');
plot(100*(X20(3,:,iloop)),'LineWidth',1.0,'color','b');
plot(100*(X30(3,:,iloop)),'LineWidth',1.0,'color','b');
plot(100*(X40(3,:,iloop)),'LineWidth',1.0,'color','b');
title('Government Spending, $G_t$','Interpreter','latex')

subplot(2,3,4)
plot(100*(irf(4,:,2)+i_ss),'LineWidth',2,'color','b');
hold on
plot(100*(X10(4,:,iloop)+i_ss),'-*','LineWidth',2.0,'color','b');
plot(100*(X20(4,:,iloop)+i_ss),'LineWidth',1.0,'color','b');
plot(100*(X30(4,:,iloop)+i_ss),'LineWidth',1.0,'color','b');
plot(100*(X40(4,:,iloop)+i_ss),'LineWidth',1.0,'color','b');
title('Nominal Interest Rate, $i_t$','Interpreter','latex')
% To ease interpretation, steady state value is added (since the u_t variables 
% are all relative to no-trap steady state values
% and thus, zero in the no-trap regime)

subplot(2,3,5)
plot(100*(irf(5,:,2)+r_ss),'LineWidth',2,'color','b');
hold on
plot(100*(X10(5,:,iloop)+r_ss),'-*','LineWidth',2.0,'color','b');
plot(100*(X20(5,:,iloop)+r_ss),'LineWidth',1.0,'color','b');
plot(100*(X30(5,:,iloop)+r_ss),'LineWidth',1.0,'color','b');
plot(100*(X40(5,:,iloop)+r_ss),'LineWidth',1.0,'color','b');
title('Real Interest Rate, $r_t$','Interpreter','latex')
% To ease interpretation, steady state value is added (since the u_t variables 
% are all relative to no-trap steady state values
% and thus, zero in the no-trap regime)

subplot(2,3,6)
plot((irf(1,:,2)-irf(1,:,1))./irf(3,:,2),'LineWidth',1.5,'color','k');
axis([0 60 0 3])
title('Fiscal Multiplier, $\partial Y_t/\partial G_t$','Interpreter','latex')

xSize = 20; ySize = 16;
set(gcf,'Units','centimeters','Position',[0 0 xSize ySize],'PaperUnits','centimeters' ...
     ,'PaperPosition',[0 0 xSize ySize],'PaperSize',[xSize-2 ySize-1],'PaperPositionMode','auto')

print -dpdf output2.pdf

