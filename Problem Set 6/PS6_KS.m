%% Code for solving a simplified version of K&S (1998)
% Quantitative Macroeconomics - IDEA programme

%% Initial values and parameters

%%%%%%%%%%%% Finding the transition matrix for the state %%%%%%%%%%%%%%

% The system of equations 
A= [ 1  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0 ; ...
     0  1  0  0  0  1  0  0  0  0  0  0  0  0  0  0 ; ...
     0  0  0  0  0  0  0  0  0  0  1  0  0  0  1  0 ; ...
     0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  1 ; ...
     0  0  0  0  0  0  0  0  1  0  0  0  1  0  0  0 ; ...
     0  0  0  0  0  0  0  0  0  1  0  0  0  1  0  0 ; ...
     0  0  1  0  0  0  1  0  0  0  0  0  0  0  0  0 ; ...
     0  0  0  1  0  0  0  1  0  0  0  0  0  0  0  0 ; ...
     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 ; ...
     0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0 ; ...
     0  0  0  0  0  0  0  0 5.6 0 -1  0  0  0  0  0 ; ...
    -1 0 28/3 0  0  0  0  0  0  0  0  0  0  0  0  0 ; ...
  .02 .48 .05 .45 0 0  0  0  0  0  0  0  0  0  0  0 ; ...
     0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0 ; ...
     0  0  0  0  0  0  0 0 .02 .48 .05 .45 0 0 0  0 ; ...
     0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0 ];
  
 
b= [7/8; 7/8; 7/8; 7/8; 1/8; 1/8; 1/8; 1/8; 7/24; 21/40; 0; 0; 0.02; 0.005; 0.05; 0.02];


pize = reshape(A^-1*b,4,4);



% transtion matrix aggregate state

piZ = [ 7/8  1/8;...
        1/8  7/8];
    
    
%%%%%%%%%%%%  Parameters   %%%%%%%%%%%%%%%%%%%%%%%%%%

betta=0.95;
delta=0.0025;
z=[1.01 0.99];
alfa=0.36;
L=[0.96, 0.9];
    
%%%%%%%%%%%%% Starting values for V %%%%%%%%%%%%%%%%%%%     
    
v1g = @(k,K) log( alfa*z(1)*(K/L(1))^(alfa-1)*k+ (1-alfa)*z(1)*(K/L(1))^(alfa) -delta*k )/(1-betta);
v1b = @(k,K) log( alfa*z(2)*(K/L(2))^(alfa-1)*k+ (1-alfa)*z(2)*(K/L(2))^(alfa) -delta*k )/(1-betta);
v0g = @(k,K) log( alfa*z(1)*(K/L(1))^(alfa-1)*k -delta*k )/(1-betta);
v0b = @(k,K) log( alfa*z(2)*(K/L(2))^(alfa-1)*k -delta*k )/(1-betta);


%%%%%%%%%%%%% Grid for k and K %%%%%%%%%%%%%%%%%%%%%%%%

k_grid=[0:0.1:5,5.3:0.3:50];
K_grid=[16:0.04:18.5];

% Evaluation of the VF
for j=1:size(K_grid,2)
V1g(:,j)= v1g(k_grid,K_grid(j))';
V1b(:,j)= v1b(k_grid,K_grid(j))';
V0g(:,j)= v0g(k_grid,K_grid(j))';
V0b(:,j)= v0b(k_grid,K_grid(j))';
end



%%
%%%%%% Perceived law of motion  %%%%%%%%%%%
% initial values

b0g=0;
b1g=1;
b0b=0;
b1b=1;

%%
for iter_b=1:1000
iter_b
% zi is the index for good shock 
H=@(K,zi) exp( (b0g+b1g*log(K))*zi+ (b0b+b1b*log(K))*(1-zi) );

% approximation

Ha= @(K,zi) min(abs(K_grid-H(K,zi)));




%% Solution of the consumer problem


% Consumption for each possible decision

% e=1 employed
% g=1 good times  =2 bad times
c= @(i,I,e,g) max(alfa*z(g)*(K_grid(I)/L(g))^(alfa-1).*k_grid(i)+ ...
             (1-alfa)*z(g)*(K_grid(I)/L(g))^(alfa)*e +(1-delta)*k_grid(i) ...
             - k_grid,0) ;

         
for iter=1:1000         
for i=1:size(k_grid,2)
    for I=1:size(K_grid,2)
        
        % approximation next period capital
        
        [dif,Ip]=min(abs(K_grid-H(K_grid(I),1)));
        V0gt(i,I)= max(log(c(i,I,0,1))' + betta * ([pize(1,:)]*([V0g(:,Ip),V1g(:,Ip),V0b(:,Ip),V1b(:,Ip)]'))');
        V1gt(i,I)= max(log(c(i,I,1,1))' + betta * ([pize(2,:)]*([V0g(:,Ip),V1g(:,Ip),V0b(:,Ip),V1b(:,Ip)]'))');  
       
        [dif,Ip]=min(abs(K_grid-H(K_grid(I),0)));
        V0bt(i,I)= max(log(c(i,I,0,2))' + betta * ([pize(3,:)]*([V0g(:,Ip),V1g(:,Ip),V0b(:,Ip),V1b(:,Ip)]'))');
        V1bt(i,I)= max(log(c(i,I,1,2))' + betta * ([pize(4,:)]*([V0g(:,Ip),V1g(:,Ip),V0b(:,Ip),V1b(:,Ip)]'))');
        
    end
    
end

dev= max(max(abs( [V0gt-V0g,V1gt-V1g,V0bt-V0b,V1bt-V1b])));

if dev<0.00001
    break
else
    V0g=V0gt;
    V1g=V1gt;
    V0b=V0bt;
    V1b=V1bt;
end 

end        
        

% Recover the policy function 


for i=1:size(k_grid,2)
    for I=1:size(K_grid,2)
        
        % approximation next period capital
        
        [dif,Ip]=min(abs(K_grid-H(K_grid(I),1)));
        [V0gt(i,I),a(i,I,2,1)]= max(log(c(i,I,0,1))' + betta * ([pize(1,:)]*([V0g(:,Ip),V1g(:,Ip),V0b(:,Ip),V1b(:,Ip)]'))');
        [V1gt(i,I),a(i,I,1,1)]= max(log(c(i,I,1,1))' + betta * ([pize(2,:)]*([V0g(:,Ip),V1g(:,Ip),V0b(:,Ip),V1b(:,Ip)]'))');  
       
        [dif,Ip]=min(abs(K_grid-H(K_grid(I),0)));
        [V0bt(i,I),a(i,I,2,2)]= max(log(c(i,I,0,2))' + betta * ([pize(3,:)]*([V0g(:,Ip),V1g(:,Ip),V0b(:,Ip),V1b(:,Ip)]'))');
        [V1bt(i,I),a(i,I,1,2)]= max(log(c(i,I,1,2))' + betta * ([pize(4,:)]*([V0g(:,Ip),V1g(:,Ip),V0b(:,Ip),V1b(:,Ip)]'))');
        
    end
    
end

%Store the policy functions. 
   assets(:,:,2,1) = a(i,I,2,1); %employed, bad
   assets(:,:,1,1)= a(i,I,1,1); % unemployed, bad
   assets(:,:,2,2) = a(i,I,2,2); % employed, good
   assets(:,:,1,2)= a(i,I,1,2); %unemployed, good

%% Simulation


% A sequence of TFP
% using the index =1 good ,  =2 bad

if iter_b==1
    
zt(1)=1;
    
for t=2:2000
    draw=rand;
    zt(t)= 1+(rand>=piZ(zt(t-1),1));
end
% Splitting the sample for good and bad times

% "burning" the first 200 periods
ztb=zt;
ztb(1:200)=0;
% Construct an index for the good times
i_zg=find(zt==1);

% Construct an index for the bad times
i_zb=find(zt==2);

% initial distribution of assets and employment
% =1 employed
N_state(1:960,:,1)=ones(960,1)*[26,1];
% =2 unemployed
N_state(961:1000,:,1)=ones(40,1)*[26,2];

K_sim(1)=find(K_grid == 17); %The index is for the generated capital from the simulated series. 

%%%%%%%%%%%%%%%%%% Simulating the Distributions %%%%%%%%%%%%%%%%%%%%%%%%%%%

for t=2:2000
for n=1:1000
    
% Evolution of assets
    N_state(n,1,t)=a(N_state(n,1,t-1),K_sim(t-1),N_state(n,2,t-1),zt(t-1));
% Evolution of the employment status     
    N_state(n,2,t)= 2-(rand>=pize(1 + zt(t-1)*2 - N_state(n,2,t-1),zt(t)*2-1)/piZ(zt(t-1),zt(t)));
    
   
end

% Storage of the sequence of aggregate capital
[dev2, K_sim(t)]=min(abs(k_grid(round(mean(N_state(:,1,t))))-K_grid));


end

else
    
    
for t=2:2000
for n=1:1000
    
% Evolution of assets
    N_state(n,1,t)=a(N_state(n,1,t-1),K_sim(t-1),N_state(n,2,t-1),zt(t-1));
   
end

% Storage of the sequence of aggregate capital
[dev2, K_sim(t)]=min(abs(k_grid(round(mean(N_state(:,1,t))))-K_grid));


end

end

% Regression model for the evolution of aggregate capital

% regression for good times (burning the first 20 periods of g times)

Yg=log(K_grid(K_sim(i_zg(20:end)))');
Xg=[ones(size(i_zg(20:end),2),1),log(K_grid(K_sim(i_zg(20:end)-1))')] ;   
Bg=Xg\Yg
b0gp=Bg(1);
b1gp=Bg(2);
% regression for bad times (burning the first 20 periods of bad times.

Yb=log(K_grid(K_sim(i_zb(20:end)))');
Xb=[ones(size(i_zb(20:end),2),1),log(K_grid(K_sim(i_zb(20:end)-1))')]  ;  
Bb=Xb\Yb
b0bp=Bb(1);
b1bp=Bb(2);


dev_b=max(abs([b0g-b0gp b1g-b1gp b0b-b0bp b1b-b1bp]))

pause(1)
if dev_b<=0.01
    break
end

b0g=0.1*b0gp+0.9*b0g;
b1g=0.1*b1gp+0.9*b1g;
b0b=0.1*b0bp+0.9*b0b;
b1b=0.1*b1bp+0.9*b1b;
 

end

%% Number of Iterations
disp('Number of Iterations')
disp(iter_b)

%% Mean of the Simulated series
mean_s = mean(K_grid(K_sim));
disp('Simulated series Aggregate/Average Capital is')
disp(mean_s)

%% R2 for the regression

mdlg = fitlm(Xg(:,2),Yg);
R2g = mdlg.Rsquared.Ordinary;
stdg= mdlg.RMSE;
disp('R2 for the good shock')
disp(R2g)

mdlb = fitlm(Xb(:,2),Yb);
R2b = mdlb.Rsquared.Ordinary;
stdb= mdlb.RMSE;
disp('R2 for the bad shock')
disp(R2b)
%% Graphs for the Policy Functions and Asset Distribution

% Evolution of the Assets distribution
%Figure
for t_ind=1:100
    
  hist(k_grid(reshape(N_state(:,1,t_ind),1,1000)),40)
  legend(num2str(t_ind))
  pause(1)
   
end

%% Statistics for the distriburion
W = k_grid(reshape(N_state(:,1,t_ind),1,1000));

mean_w = mean(W);
disp('Mean of Asset Distribution')
disp(mean_w)

std_w = std(W);
disp('Standard Deviation of Asset Distribution')
disp(std_w)

skew_w=skewness(W);
disp('Skewness of Asset Distribution');
disp(skew_w)

%% Retrieve the Policy function from the indices. 

G0g = k_grid(a(:,26,2,1)); % unemployed, good
G1g = k_grid(a(:,26,1,1)); % employed, good
G0b = k_grid(a(:,26,2,2)); % unemployed, bad
G1b = k_grid(a(:,26,1,2)); % employed, bad

figure(1)
plot(k_grid, G1b,'b' )
hold on
plot(k_grid, G0b, 'r' )
hold on
plot(k_grid,k_grid , 'y--')
title ('Asset policy function for the bad state')
legend('Employed', 'Unemployed', '45^0')
xlabel('k today ')
ylabel('k tomorrow')
hold off

figure(2)
plot(k_grid, G1g, 'b' )
hold on
plot(k_grid, G0g, 'r' )
hold on
plot(k_grid,k_grid , 'y--')
title ('Asset policy function for the good state')
legend('Employed', 'Unemployed', '45^0')
xlabel('k today ')
ylabel('k tomorrow')
hold off

%% Employment distribution

for t_ind=1:100  
    hist(reshape(N_state(:,2,t_ind),1,1000),40)
    title('Employment Distribution for 1000 individuals after 100 periods ')
    xticks([1 2])
    xticklabels({'Employed','Unemployed'})
end
print -dpdf histe_fig6.eps
%% Compare the asset distribution in equilibrium after 7 periods of being in a bad state (low z) as opposed
%to being in the high state.

grouped_b = mat2cell( i_zb, 1, diff( [0, find(diff(i_zb) ~= 1), length(i_zb)] )) ;
    
for i=1:size(grouped_b,2)
    if size(grouped_b{i},2)==7
     g7b = grouped_b{i};
     break
    end
end

for t=g7b(1):g7b(7)
hist(k_grid(reshape(N_state(:,1,t),1,1000)),40);
title('Capital Distribution for 1000 individuals after 7 consecutive bad periods ')
end
print -dpdf hist7b_fig7.eps

W1=(k_grid(reshape(N_state(:,1,t),1,1000)))
mean_w1 = mean(W1);
disp('Mean of Asset Distribution')
disp(mean_w1)

std_w1 = std(W1);
disp('Standard Deviation of Asset Distribution')
disp(std_w1)

skew_w1 =skewness(W1);
disp('Skewness of Asset Distribution');
disp(skew_w1)

%% For good 
grouped_g = mat2cell( i_zg, 1, diff( [0, find(diff(i_zg) ~= 1), length(i_zg)] )) ;
    
for i=1:size(grouped_g,2)
    if size(grouped_g{i},2)==7
     g7g = grouped_g{i};
     break
    end
end


for t=g7g(1):g7g(7)
hist(k_grid(reshape(N_state(:,1,t),1,1000)),40);
title('Capital Distribution for 1000 individuals after 7 consecutive good periods ')
end
print -dpdf hist7g_fig8.eps

W2=(k_grid(reshape(N_state(:,1,t),1,1000)));
mean_w2 = mean(W1);
disp('Mean of Asset Distribution')
disp(mean_w2)

std_w2 = std(W2);
disp('Standard Deviation of Asset Distribution')
disp(std_w2)

skew_w2 =skewness(W2);
disp('Skewness of Asset Distribution');
disp(skew_w2)
