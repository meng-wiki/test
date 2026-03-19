% Huasheng Xie, huashengxie@gmail.com, 2026-01-15 10:23
% Code: VEQ, Variational EQuilibrium solver.
% Solve plasma equilibrium with fast variational method,
% PDE->algebraic equations, with error to ~<5%.
% This version: 2D, MHD, fixed boundary, limiter, MXH configuration
% Ref: Haney1988, Haney1995, Lao1981, Arbon2021, Xie2016
% Ack.: Li Yue-yan, Deepseek, Gemini, etc
% Use COCOS=1, with 2*5+1*4+1=15 profile parameters
% 
% Keep to use rho=rho_geom, not rho_psi.
% 26-01-16 23:13 modify to 3-moment, higher radial order, 12 pars, 
% test agree well with GS.m

clear; clc; close all;

mu0=4e-7*pi;
parin.mu0=mu0;
% input parameters
icase=3;
if(icase==1)
  R0=0.4; % m
  a=0.13; % m
  A=R0/a;
  Ip=0.01e6; % A
  B0=0.70; % T

  ka=1.0;
  da=0.0;
  alpha_p=5.0;
  alpha_f=3.32;
  beta0=1.0;
elseif(icase==2)
  R0=4.7; % m
  a=1.7; % m
  A=R0/a;
  Ip=19.0e6; % A
  B0=4.5; % T

  ka=2.0;
  da=0.40;
  alpha_p=3.32;
  alpha_f=3.32;
  beta0=0.8;
elseif(icase==3)
  R0=1.05; % m
  a=R0/1.85; % m
  A=R0/a;
  Ip=3.0e6; % A
  B0=3.0; % T

  ka=2.2;
  da=0.5;
  alpha_p=5.0;
  alpha_f=3.32;
  beta0=1.5;

  % R0=1.05; a=R0/1.85; ka=1.8; da=-0.3; B0=2.0;
end

s1a=asin(da);
s2a=0.;
s3a=-0.0;
c0a=0.0;
c1a=-0.0;
Z0=0;

iprof=1;
parin.R0=R0;
parin.Z0=Z0;
parin.B0=B0;
parin.Ip=Ip;
parin.a=a;
parin.ka=ka;
parin.s1a=s1a;
parin.s2a=s2a;
parin.s3a=s3a;
parin.c0a=c0a;
parin.c1a=c1a;

parin.beta0=beta0;
parin.alpha_p=alpha_p;
parin.alpha_f=alpha_f;

% flux coordinate grids
N_rho = 2^3;       % radial
N_theta = 12;     % theta
[rho_vec,theta_vec,rho,theta,rho_wi,theta_wi]=gen_grid(N_rho,N_theta,1,1);
%%

if(1==1)
  tic;
  % initial guess of variational parameters [nu, sigma, eta, kappa0,nu2]
  % x0 = [1,1,0,0];
  x0=zeros(1,12);
  % x0(7)=1;

  cputime0=cputime;
  it=1;
  maxdx=1.0;
  flag=1;

  while(maxdx>=10e-3)
    % if(maxdx<=1e-2 && flag==1)
    %   N_rho = 2^5;       % radial
    %   N_theta = 2^5     % theta
    %   [~,~,rho,theta,rho_wi,theta_wi]=gen_grid(N_rho,N_theta,1,1);
    %   flag=0;
    % end
    %     N_rho=40; N_theta=50;
    % [rho_vec,theta_vec,rho,theta,rho_wi,theta_wi]=gen_grid(N_rho,N_theta,0,0);
    [C0,psi0] = calC0psi0(x0,rho,theta,rho_wi,theta_wi,parin)
    % C0=4.0; psi0=0.3;

    options = optimoptions('fsolve','Display','iter','Algorithm','levenberg-marquardt','FunctionTolerance',1e-4,'StepTolerance',1e-4);
    parvar = fsolve(@(par) Ls_fun(par,rho,theta,rho_wi, theta_wi,...
      parin,C0,psi0), x0,options);
    % 1.5111    1.8696    0.2972    0.3224   -0.4851 % old fmin
    % 1.5214    1.8837    0.3332    0.3235   -0.5062 % new fsolve, with bug
    % 1.5154    1.8707    0.2997    0.3225   -0.4945 % new fsolve, fix bug

    maxdx=max(abs(x0-parvar));
    x0=parvar
    it=it+1
  end
  cputime1=cputime;
  runtime=cputime1-cputime0
  runtime2=toc

  %% plot
  N_rho = 100;       % radial
  N_theta = 150*1;     % theta
  % parvar=[1.5115    0.3224    0.2968    1.8694   -0.4860];
  [rho_vec,theta_vec,rho,theta,rho_wi,theta_wi]=gen_grid(N_rho,N_theta,0,0);
  [R,Z,J,R_ro,R_th,Z_ro,Z_th,R_roro,R_roth,...
  R_thth,Z_roro,Z_roth,Z_thth,J_ro,J_th,...
  psi,psi_ro,psi_roro,hp,hf,hp_prime,hf_prime,...
  R_h0,R_h1,Z_k0,Z_k1,Z_v0,Z_v1,R_s10,R_s11,R_s20,R_s30,R_c00,R_c10,...
  psi_R,psi_Z,psi_nu0,psi_nu1,psi_nu2] = ...
  shape_fun(rho,theta,parvar,parin);
  prof = p_fun(rho,theta,rho_vec,theta_vec,rho_wi, theta_wi, ...
    parvar,parin,C0,psi0);

  hfg=figure('unit','normalized','Position',[0.01 0.05 0.6 0.75],...
    'DefaultAxesFontSize',12);

  subplot(331);
  plot(prof.rhon_vec,prof.p_vec,'LineWidth',2);
  xlabel('\rho'); ylabel('p');
  title(['R_0=',num2str(R0),', a=',num2str(a),', B_0=',num2str(B0)]);
  subplot(332);
  plot(prof.rhon_vec,prof.F_vec,'LineWidth',2);
  xlabel('\rho'); ylabel('F');
  title(['\beta_0=',num2str(beta0),', q_0=',num2str(prof.q_vec(1),3), ...
    ', \kappa=',num2str(ka)]);
  subplot(333);
  plot(prof.rhon_vec,prof.q_vec,'LineWidth',2);
  xlabel('\rho'); ylabel('q');
  title(['\delta=',num2str(da),', I_p=',num2str(Ip,3), ...
    ', \psi_0=',num2str(psi0,3),', C_0=',num2str(C0,3)]);
  subplot(334);
  contour(R,Z,prof.BR,50,'LineWidth',1.5);  axis equal; title('B_R');
  xlabel('R (m)'); ylabel('Z (m)');
  title(['B_R, \alpha_p=',num2str(alpha_p), ...
    ', \alpha_f=',num2str(alpha_f,3)]);
  hold on; plot(R(:,end),Z(:,end),'k-');
  subplot(335);
  contour(R,Z,prof.BZ,50,'LineWidth',1.5);  axis equal; title('B_Z');
  xlabel('R (m)'); ylabel('Z (m)');
  title(['B_Z, V_p=',num2str(prof.Vp,3),', \beta_t=', ...
    num2str(prof.betat,3),', \beta_p=',num2str(prof.betap,3)]);
  hold on; plot(R(:,end),Z(:,end),'k-');
  subplot(337);
  contour(R,Z,prof.Bp,50,'LineWidth',1.5);  axis equal;
  title(['B_p, l_i=',num2str(prof.li,3),', B_{pa}=', ...
    num2str(prof.Bpa,3),', S_p=',num2str(prof.Sp,3)]);
  xlabel('R (m)'); ylabel('Z (m)');
  hold on; plot(R(:,end),Z(:,end),'k-');
  subplot(338);
  contour(R,Z,prof.jphi,50,'LineWidth',1.5);  axis equal;
  title('j_\phi');
  xlabel('R (m)'); ylabel('Z (m)');
  hold on; plot(R(:,end),Z(:,end),'k-');

  subplot(3,3,[6,9]);
  contour(R,Z,psi, 20,'LineWidth',1.5);
  title(['R_m=',num2str(R(1,1),3),', \beta_N=',num2str(prof.betaN,3), ...
    ', L_p=',num2str(prof.Lp,3)]);
  xlabel(['R (m), runtime=',num2str(runtime,3),'s']);
  ylabel(['Z (m), parvar=',num2str(parvar,3)]);
  axis equal;  colorbar;
  hold on; plot(R(:,end),Z(:,end),'k-');

  print('-dpng',['veq_mxh_fsolve_plot_R0=',num2str(R0),...
    ',a=',num2str(a),',k=',num2str(ka),',d=',num2str(da), ...
    ',B0=',num2str(B0),',Ip=',num2str(Ip), ...
    ',iprof=',num2str(iprof), ...
    ',ap=',num2str(alpha_p),',af=',num2str(alpha_f),...
    ',icase=',num2str(icase),'_3m_12par.png']);

  %%
  if(1==0)
    figure;plot(theta(:,end),prof.Bp(:,end),theta(:,end), ...
      prof.BR(:,end),theta(:,end),prof.BZ(:,end),'LineWidth',2);
    legend('B_p','B_R','B_Z'); xlabel('\theta');xlim([0,pi]);
  end
  if(1==0)
    figure;plot(rho_vec,prof.Bpavg_vec);
  end
  %%

end

function [rho_vec,theta_vec,rho, theta,rho_wi,theta_wi]=gen_grid( ...
  N_rho,N_theta,irnode,itnode)
if(irnode==1) % Gauss nodes
  [rho_vec,rho_wi]=calgaussnodes(N_rho);
else
  rho_vec = linspace(1e-5, 0.99999, N_rho);
  rho_wi=(rho_vec(2)-rho_vec(1))+0.*rho_vec;
  % rho_vec = linspace(0, 1, N_rho+1); % 25-03-07 10:54 change to midpoint
  % rho_vec=0.5*(rho_vec(1:(end-1))+rho_vec(2:end));
  % rho_wi=1/N_rho+0.*rho_vec;
end
rho_wi=repmat(rho_wi,[N_theta,1]);
if(itnode==1) % Gauss nodes
  [theta_vec,theta_wi]=calgaussnodes(N_theta);
  theta_vec=2*pi*theta_vec.';
  theta_wi=theta_wi*2*pi;
else
  theta_vec = linspace(0, 2*pi, N_theta);
  theta_wi=(theta_vec(2)-theta_vec(1))+0.*theta_vec;
  % theta_vec = linspace(0, 2*pi, N_theta+1);
  % theta_vec=0.5*(theta_vec(1:(end-1))+theta_vec(2:end));
  % theta_wi=2*pi/N_theta+0.*theta_vec;
end
theta_wi=repmat(theta_wi.',[1,N_rho]);
[rho, theta] = meshgrid(rho_vec, theta_vec);
end

function [xi,wi]=calgaussnodes(n)

if(n==4) % Gauss integral nodes xi and weight wi for [0,1]
  xi=[0.0694318442029737	0.330009478207572	0.669990521792428	0.930568155797026];
  wi=[0.173927422568728	0.326072577431274	0.326072577431274	0.173927422568728];
elseif(n==6)
  xi=[0.0337652428984240	0.169395306766868	0.380690406958402	0.619309593041599	0.830604693233132	0.966234757101576];
  wi=[0.0856622461895866	0.180380786524070	0.233956967286346	0.233956967286346	0.180380786524070	0.0856622461895866];
elseif(n==8)
  xi=[0.0198550717512319	0.101666761293187	0.237233795041836	0.408282678752175	0.591717321247825	0.762766204958165	0.898333238706813	0.980144928248768];
  wi=[0.0506142681451894	0.111190517226688	0.156853322938944	0.181341891689181	0.181341891689181	0.156853322938944	0.111190517226688	0.0506142681451894];
elseif(n==10)
  xi=[0.0130467357414142	0.0674683166555077	0.160295215850488	0.283302302935376	0.425562830509184	0.574437169490816	0.716697697064624	0.839704784149512	0.932531683344492	0.986953264258586];
  wi=[0.0333356721543452	0.0747256745752909	0.109543181257991	0.134633359654998	0.147762112357377	0.147762112357377	0.134633359654998	0.109543181257991	0.0747256745752909	0.0333356721543452];
elseif(n==12)
  xi=[0.00921968287664038	0.0479413718147625	0.115048662902848	0.206341022856691	0.316084250500910	0.437383295744266	0.562616704255734	0.683915749499090	0.793658977143309	0.884951337097152	0.952058628185238	0.990780317123360];
  wi=[0.0235876681932572	0.0534696629976599	0.0800391642716736	0.101583713361533	0.116746268269178	0.124573522906702	0.124573522906702	0.116746268269178	0.101583713361533	0.0800391642716736	0.0534696629976599	0.0235876681932572];
elseif(n==16)
  xi=[0.00529953250417503	0.0277124884633837	0.0671843988060841	0.122297795822499	0.191061877798678	0.270991611171386	0.359198224610371	0.452493745081181	0.547506254918819	0.640801775389630	0.729008388828614	0.808938122201322	0.877702204177502	0.932815601193916	0.972287511536616	0.994700467495825];
  wi=[0.0135762297058783	0.0311267619693245	0.0475792558412468	0.0623144856277673	0.0747979944082887	0.0845782596975015	0.0913017075224620	0.0947253052275344	0.0947253052275344	0.0913017075224620	0.0845782596975015	0.0747979944082887	0.0623144856277673	0.0475792558412468	0.0311267619693245	0.0135762297058783];
elseif(n==32)
  xi=[0.00136806907525922	0.00719424422736581	0.0176188722062468	0.0325469620311302	0.0518394221169740	0.0753161931337150	0.102758102016029	0.133908940629855	0.168477866534892	0.206142121379619	0.246550045533885	0.289324361934682	0.334065698858936	0.380356318873931	0.427764019208602	0.475846167156131	0.524153832843869	0.572235980791398	0.619643681126069	0.665934301141064	0.710675638065318	0.753449954466115	0.793857878620381	0.831522133465108	0.866091059370145	0.897241897983971	0.924683806866285	0.948160577883026	0.967453037968870	0.982381127793753	0.992805755772634	0.998631930924741];
  wi=[0.00350930500473635	0.00813719736545343	0.0126960326546314	0.0171369314565110	0.0214179490111136	0.0254990296311883	0.0293420467392679	0.0329111113881811	0.0361728970544244	0.0390969478935353	0.0416559621134735	0.0438260465022020	0.0455869393478821	0.0469221995404024	0.0478193600396375	0.0482700442573640	0.0482700442573640	0.0478193600396375	0.0469221995404024	0.0455869393478821	0.0438260465022020	0.0416559621134735	0.0390969478935353	0.0361728970544244	0.0329111113881811	0.0293420467392679	0.0254990296311883	0.0214179490111136	0.0171369314565110	0.0126960326546314	0.00813719736545343	0.00350930500473635];
end
end

% shape and profile function
% function [R,Z,J,R_ro,R_th,Z_ro,Z_th,psi,psi_ro,psi_theta, ...
%   hp,hf,hp_prime,hf_prime,grad_psi_R,grad_psi_Z,grad_psi_sq] = ...
%   shape_fun(rho,theta,parvar,parin)
function [R,Z,J,R_ro,R_th,Z_ro,Z_th,R_roro,R_roth,...
  R_thth,Z_roro,Z_roth,Z_thth,J_ro,J_th,...
  psi,psi_ro,psi_roro,hp,hf,hp_prime,hf_prime,...
  R_h0,R_h1,R_s10,R_s11,R_s20,R_s30,R_c00,R_c10,Z_k0,Z_k1,Z_v0,Z_v1,R_h2,R_s12,Z_k2,...
  psi_R,psi_Z,psi_nu0,psi_nu1,psi_nu2] = ...
  shape_fun(rho,theta,parvar,parin)

% variational parameters [h0,h1,k0,k1,s10,s11,nu0,nu1,s20,s30,c00,c10,v0,v1],psia
h0=parvar(1);
h1=parvar(2);
k0=parvar(3);
k1=parvar(4);
s10=parvar(5);
s11=parvar(6);
nu0=parvar(7);
nu1=parvar(8);
% s20=parvar(9);
% s30=parvar(10);
% c00=parvar(11);
% c10=parvar(12);
% v0=parvar(13);
% v1=parvar(14);
nu2=parvar(9);
h2=parvar(10);
k2=parvar(11);
s12=parvar(12);

s20=0;
s30=0;
c00=0;
c10=0;
v0=0;
v1=0;

R0=parin.R0;
Z0=parin.Z0;
a=parin.a;
ka=parin.ka; % ka
s1a=parin.s1a;
s2a=parin.s2a;
s3a=parin.s3a;
c0a=parin.c0a;
c1a=parin.c1a;
rho2=rho.^2; rho3=rho.^3; rho4=rho.^4; rho5=rho.^5; rho6=rho.^6;

fac0=(1-rho2); fac0_ro=-2*rho; fac0_roro=-2-0*rho;
fac1=rho.*(1-rho2); fac1_ro=1-3*rho2; fac1_roro=-6*rho;
fac2=rho2.*(1-rho2); fac2_ro=2*rho-4*rho3; fac2_roro=2-12*rho2;
u1=(2*rho2-1); u2=2*(2*rho2-1).^2-1;
u1_ro=4*rho; %u1_roro=2+0.*rho;
u1_roro=4+0.*rho;
u2_ro=4*(2*rho2-1).*(4*rho); u2_roro=16*(6*rho2-1);

psi = rho2+fac2.*(nu0+nu1*u1+nu2*u2);
psi_ro =2*rho+fac2_ro.*(nu0+nu1*u1+nu2*u2)+fac2.*(nu1*u1_ro+nu2*u2_ro);
psi_roro =2+fac2_roro.*(nu0+nu1*u1+nu2*u2)+2*fac2_ro.*(nu1*u1_ro+nu2*u2_ro)+fac2.*(nu1*u1_roro+nu2*u2_roro);
% psi = rho2.*(1+(1-rho2).*(nu0+nu1*(2*rho2-1)+nu2*(2*(2*rho2-1).^2-1))); % 26-01-16 16:06
psi_nu0=fac2;
psi_nu1=fac2.*u1;
psi_nu2=fac2.*u2;

h=fac0.*(h0+h1*u1+h2*u2);
h_ro=h0*fac0_ro+h1*(fac0_ro.*u1+fac0.*u1_ro)+h2*(fac0_ro.*u2+fac0.*u2_ro);
h_roro=h0*fac0_roro+h1*(fac0_roro.*u1+2*fac0_ro.*u1_ro+fac0.*u1_roro)+...
  h2*(fac0_roro.*u2+2*fac0_ro.*u2_ro+fac0.*u2_roro);
h_h0=fac0;
h_h1=fac0.*u1;
h_h2=fac0.*u2;

k=ka+fac0.*(k0+k1*u1+k2*u2);
k_ro=k0*fac0_ro+k1*(fac0_ro.*u1+fac0.*u1_ro)+k2*(fac0_ro.*u2+fac0.*u2_ro);
k_roro=k0*fac0_roro+k1*(fac0_roro.*u1+2*fac0_ro.*u1_ro+fac0.*u1_roro)+...
  k2*(fac0_roro.*u2+2*fac0_ro.*u2_ro+fac0.*u2_roro);
k_k0=fac0;
k_k1=fac0.*u1;
k_k2=fac0.*u2;

% k=ka+(1-rho2).*(k0+k1*(2*rho2-1));
% k_ro=-k0*2*rho+k1*(6*rho-8*rho3);
% k_roro=-2*k0+k1*(6-24*rho2);
% k_k0=(1-rho2);
% k_k1=(1-rho2).*(2*rho2-1);

% s1=rho.*(s1a+(1-rho2).*(s10+s11*(2*rho2-1)));
% s1_ro=(s1a+(1-rho2).*(s10+s11*(2*rho2-1)))+rho.*(-s10*2*rho+s11*(6*rho-8*rho3));
% s1_roro=2*(-s10*2*rho+s11*(6*rho-8*rho3))+rho.*(-s10*2+s11*(6-24*rho2));
% s1_s10=rho.*(1-rho2);
% s1_s11=rho.*(1-rho2).*(2*rho2-1);

s1=rho*s1a+fac1.*(s10+s11*u1+s12*u2);
s1_ro=s1a+s10*fac1_ro+s11*(fac1_ro.*u1+fac1.*u1_ro)+s12*(fac1_ro.*u2+fac1.*u2_ro);
s1_roro=s10*fac1_roro+s11*(fac1_roro.*u1+2*fac1_ro.*u1_ro+fac1.*u1_roro)+...
  s12*(fac1_roro.*u2+2*fac1_ro.*u2_ro+fac1.*u2_roro);
s1_s10=fac1;
s1_s11=fac1.*u1;
s1_s12=fac1.*u2;


s2=rho2.*(s2a+s20*(1-rho2));
s2_ro=s2a*2*rho+s20*(2*rho-4*rho3);
s2_roro=s2a*2+s20*(2-12*rho2);
s2_s20=rho2.*(1-rho2);

s3=rho2.*(s3a+s30*(1-rho2));
s3_ro=s3a*2*rho+s30*(2*rho-4*rho3);
s3_roro=s3a*2+s30*(2-12*rho2);
s3_s30=rho2.*(1-rho2);

c0=c0a+c00*(1-rho2);
c0_ro=-c00*2*rho;
c0_roro=-c00*2+0*rho;
c0_c00=(1-rho2);

c1=rho.*(c1a+c10*(1-rho2));
c1_ro=c1a+c10*(1-3*rho2);
c1_roro=-c10*6*rho;
c1_c10=rho.*(1-rho2);

v=(1-rho2).*(v0+v1*(2*rho2-1));
v_ro=-v0*2*rho+v1*(6*rho-8*rho3);
v_roro=-2*v0+v1*(6-24*rho2);
v_v0=(1-rho2);
v_v1=(1-rho2).*(2*rho2-1);

th=theta;
tb=th+c0+c1.*cos(th)+s1.*sin(th)+s2.*sin(2*th)+s3.*sin(3*th);
R = R0 + h + a*rho.*cos(tb);
Z = Z0 + v - a*k.*rho.*sin(th);
tb_ro=c0_ro+c1_ro.*cos(th)+s1_ro.*sin(th)+s2_ro.*sin(2*th)+s3_ro.*sin(3*th);
tb_th=1-c1.*sin(th)+s1.*cos(th)+2*s2.*cos(2*th)+3*s3.*cos(3*th);

% to add tb_roth, tb_thth, tb_roro
tb_roth=-c1_ro.*sin(th)+s1_ro.*cos(th)+2*s2_ro.*cos(2*th)+3*s3_ro.*cos(3*th);
tb_thth=-c1.*cos(th)-s1.*sin(th)-4*s2.*sin(2*th)-9*s3.*sin(3*th);
tb_roro=c0_roro+c1_roro.*cos(th)+s1_roro.*sin(th)+s2_roro.*sin(2*th)+s3_roro.*sin(3*th);
% calculate analytical derivite
R_ro=h_ro+a*cos(tb)-a*rho.*sin(tb).*tb_ro;
R_th=-a*rho.*sin(tb).*tb_th;
Z_ro=v_ro-a*(rho.*k_ro+k).*sin(th);
Z_th=-a*k.*rho.*cos(th);
R_roro=h_roro-2*a*sin(tb).*tb_ro-a*rho.*cos(tb).*tb_ro.^2-a*rho.*sin(tb).*tb_roro;
R_roth=-a*sin(tb).*tb_th-a*rho.*cos(tb).*tb_th.*tb_ro-a*rho.*sin(tb).*tb_roth;
R_thth=-a*rho.*cos(tb).*tb_th.^2-a*rho.*sin(tb).*tb_thth;
Z_roro=v_roro-a*(k_roro.*rho+2*k_ro).*sin(th);
Z_roth=-a*(k_ro.*rho+k).*cos(th);
Z_thth=a*k.*rho.*sin(th);

% R_h0, R_h1, R_, Z_v0, Z_v1, ...
R_h0=h_h0;
R_h1=h_h1;
R_s10=-a*rho.*sin(tb).*(s1_s10.*sin(th));
R_s11=-a*rho.*sin(tb).*(s1_s11.*sin(th));
R_s20=-a*rho.*sin(tb).*(s2_s20.*sin(2*th));
R_s30=-a*rho.*sin(tb).*(s3_s30.*sin(3*th));
R_c00=-a*rho.*sin(tb).*(c0_c00);
R_c10=-a*rho.*sin(tb).*(c1_c10.*cos(th));

Z_v0=v_v0;
Z_v1=v_v1;
Z_k0=-a*k_k0.*rho.*sin(th);
Z_k1=-a*k_k1.*rho.*sin(th);


R_h2=h_h2;
R_s12=-a*rho.*sin(tb).*(s1_s12.*sin(th));
Z_k2=-a*k_k2.*rho.*sin(th);

% Jacobian
J=-(R_ro.*Z_th-R_th.*Z_ro);
epsJ=1e-15;
% J(abs(J) < epsJ) = epsJ;  % avoid singularity
J(J < epsJ) = epsJ;  % avoid singularity
J_ro=-(R_roro.*Z_th-R_roth.*Z_ro+R_ro.*Z_roth-R_th.*Z_roro);
J_th=-(R_roth.*Z_th-R_thth.*Z_ro+R_ro.*Z_thth-R_th.*Z_roth);

% psi = nu0*rho.^2 +nu1*rho.^4+ (1 - nu0-nu1)*rho.^6; % 25-02-24 10:00
% psi_ro = (2*nu0*rho +4*nu1*rho.^3+ 6*(1 - nu0-nu1)*rho.^5);
% psi_roro = (2*nu0 +12*nu1*rho.^2+ 30*(1 - nu0-nu1)*rho.^4);
% psi_nu0=rho.^2-rho.^6;
% psi_nu1=rho.^4-rho.^6;

% psi = rho2.*(1+(1-rho2).*(nu0+nu1*(2*rho2-1))); % 26-01-16 10:30
% psi_ro = 2*rho+nu0*(2*rho-4*rho3)+nu1*(-2*rho+12*rho3-12*rho5);
% psi_roro = 2+nu0*(2-12*rho2)+nu1*(-2+36*rho2-60*rho4);
% psi_nu0=rho2.*((1-rho2));
% psi_nu1=rho2.*((1-rho2).*(2*rho2-1));


% psi_theta = 0.*rho; % psi independ with theta

psi = min(max(psi, 0), 1);  % keep psi in [0,1]
iprof=1;
if(iprof==1) % exp

  hp_prime = parin.alpha_p*(exp(parin.alpha_p*psi) ...
    -exp(parin.alpha_p)) ./ (1+exp(parin.alpha_p)*(parin.alpha_p-1));
  hp=(exp(parin.alpha_p)*(parin.alpha_p*(1-psi)-1)+exp( ...
    parin.alpha_p*psi)) ./ (1+exp(parin.alpha_p)*(parin.alpha_p-1));
  hf_prime = parin.alpha_f*(exp(parin.alpha_f*psi) ...
    -exp(parin.alpha_f)) ./ (1+exp(parin.alpha_f)*(parin.alpha_f-1));
  hf=(exp(parin.alpha_f)*(parin.alpha_f*(1-psi)-1)+exp( ...
    parin.alpha_f*psi)) ./ (1+exp(parin.alpha_f)*(parin.alpha_f-1));

elseif(iprof==2) % GAQ, to do
  hp_prime = (1 - psi);
  hf_prime = 0;
  hp=0;
  hf=0;
end

% calculate grad ψ
psi_R = (Z_th.*psi_ro ) ./ J;
psi_Z = ( - R_th.*psi_ro) ./ J;
% grad_psi_sq = (Z_th.^2+R_th.^2).*psi_ro.^2./J.^2;
end

% L function
function Ls = Ls_fun(parvar,rho,theta,rho_wi, theta_wi,parin,C0,psi0)

% mu0=4e-7*pi;%mu0=parin.mu0;
[R,Z,J,R_ro,R_th,Z_ro,Z_th,R_roro,R_roth,...
  R_thth,Z_roro,Z_roth,Z_thth,J_ro,J_th,...
  psi,psi_ro,psi_roro,hp,hf,hp_prime,hf_prime,...
  R_h0,R_h1,R_s10,R_s11,R_s20,R_s30,R_c00,R_c10,Z_k0,Z_k1,Z_v0,Z_v1,R_h2,R_s12,Z_k2,...
  psi_R,psi_Z,psi_nu0,psi_nu1,psi_nu2] = ...
  shape_fun(rho,theta,parvar,parin);

p_psi=C0*parin.beta0*(1/parin.R0).*hp_prime;
FF_psi=C0*(1 - parin.beta0)*(parin.R0./1).*hf_prime;
L_term1 = psi0*(psi_roro.*(R_th.^2+Z_th.^2)./(J.*R)+...
  psi_ro.*((2*(R_th.*R_roth+Z_th.*Z_roth)-...
  (R_thth.*R_ro+Z_roth.*Z_th+R_th.*R_roth+ ...
  Z_ro.*Z_thth))./(J.*R)-((R_th.^2+Z_th.^2).*(J_ro.*R+ ...
  J.*R_ro)-(R_th.*R_ro+Z_ro.*Z_th).*(J_th.*R+ ...
  J.*R_th))./(J.*R).^2)); % .*psi_ro
L_term2 = J./R.*FF_psi;
L_term3 = J.*R.*p_psi;

G=1e2*(L_term1 + L_term2 + L_term3);

% Ls(4)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_d2+psi_Z.*Z_d2))));
Ls(1)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_nu0))));
Ls(2)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_nu1))));
Ls(3)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_h0))));
Ls(4)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_h1))));
Ls(5)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_s10))));
Ls(6)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_s11))));

% Ls(7)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_s20))));
% Ls(8)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_s30))));
% Ls(9)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_c00))));
% Ls(10)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_c10))));
Ls(7)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_Z.*Z_k0))));
Ls(8)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_Z.*Z_k1))));
% Ls(13)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_Z.*Z_v0))));
% Ls(14)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_Z.*Z_v1))));
Ls(9)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_nu2))));

Ls(10)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_h2))));
Ls(11)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_R.*R_s12))));
Ls(12)=sum(sum(rho_wi.*theta_wi.*(G.*(psi_Z.*Z_k2))));

% Ls1=1e2*sum(sum(rho_wi.*theta_wi.*((L_term1+L_term2+L_term3)).^2)); %

end


function [C0,psi0] = calC0psi0(parvar,rho,theta,rho_wi, theta_wi,...
  parin)

mu0=4e-7*pi;
[R,Z,J,R_ro,R_th,Z_ro,Z_th,R_roro,R_roth,...
  R_thth,Z_roro,Z_roth,Z_thth,J_ro,J_th,...
  psi,psi_ro,psi_roro,hp,hf,hp_prime,hf_prime,...
  R_h0,R_h1,R_s10,R_s11,R_s20,R_s30,R_c00,R_c10,Z_k0,Z_k1,Z_v0,Z_v1,R_h2,R_s12,Z_k2,...
  psi_R,psi_Z,psi_nu0,psi_nu1,psi_nu2] = ...
  shape_fun(rho,theta,parvar,parin);

% grad_psi_R = (Z_th.*psi_ro ) ./ J;
% grad_psi_Z = ( - R_th.*psi_ro) ./ J;
grad_psi_sq = (Z_th.^2+R_th.^2).*psi_ro.^2./J.^2;

% integral
integrand_numerator = (parin.beta0*(R/parin.R0).*hp_prime + ...
  (1 - parin.beta0)*(parin.R0./R).*hf_prime) .* (1-psi) .* J;
integrand_denominator = (1./R) .* grad_psi_sq .* J;
% numerator = trapz(theta_vec, trapz(rho_vec, integrand_numerator, 2));
% denominator = trapz(theta_vec, trapz(rho_vec, integrand_denominator, 2));
numerator = sum(sum(rho_wi.*theta_wi.*( integrand_numerator))); %
denominator = sum(sum(rho_wi.*theta_wi.*( integrand_denominator))); %

integrand_C = (parin.beta0*(R/parin.R0).*hp_prime + ...
  (1 - parin.beta0)*(parin.R0./R).*hf_prime) .* J;
% C_integral = trapz(theta_vec, trapz(rho_vec, integrand_C, 2));
C_integral =sum(sum(rho_wi.*theta_wi.*( integrand_C))); %
C0 = -(mu0*parin.Ip) / C_integral;

psi0 = -C0 * numerator / denominator;

end

% calculate B, q(psi), etc
function prof = p_fun(rho,theta,rho_vec,theta_vec,rho_wi, theta_wi, ...
  parvar,parin,C0,psi0) % to update for more output
% to update for different type of average, volume, surface, line, dV, etc
% to ref: Jardin10
% 25-03-07 11:25 to update the integral
% radial only support linspace grid, not Gauss-nodes grid

mu0=4e-7*pi;
[R,Z,J,R_ro,R_th,Z_ro,Z_th,R_roro,R_roth,...
  R_thth,Z_roro,Z_roth,Z_thth,J_ro,J_th,...
  psi,psi_ro,psi_roro,hp,hf,hp_prime,hf_prime,...
  R_h0,R_h1,R_s10,R_s11,R_s20,R_s30,R_c00,R_c10,Z_k0,Z_k1,Z_v0,Z_v1,R_h2,R_s12,Z_k2,...
  psi_R,psi_Z,psi_nu0,psi_nu1,psi_nu2] = ...
  shape_fun(rho,theta,parvar,parin);


grad_psi_R = (Z_th.*psi_ro ) ./ J;
grad_psi_Z = ( - R_th.*psi_ro) ./ J;
grad_psi_sq = (Z_th.^2+R_th.^2).*psi_ro.^2./J.^2;

prof.C0=C0;
prof.psi0=psi0;
prof.R=R;
prof.Z=Z;
prof.rhon=rho;
prof.theta=theta;
prof.psi=psi*psi0;
prof.psin=psi;
prof.p0=C0*psi0*parin.beta0/(mu0*parin.R0);
prof.rhon_vec=rho_vec;
prof.theta_vec=theta_vec;
prof.psin_vec=psi(1,:);
prof.psi_vec=prof.psin_vec*psi0;
prof.psi_rho_vec=psi0*psi_ro(1,:);
prof.p=prof.p0*hp;
prof.jphi=C0/mu0*(parin.beta0/parin.R0*R.*hp_prime+ ...
  (1-parin.beta0)*parin.R0./R.*hf_prime);
prof.p_vec=prof.p0*hp(1,:);
prof.BR=R_th./(R.*J).*(psi0*psi_ro);
prof.BZ=Z_th./(R.*J).*(psi0*psi_ro);
prof.Bp=sqrt(prof.BR.^2+prof.BZ.^2);
prof.F_vec=sqrt(parin.R0^2*parin.B0^2+ ...
  2*C0*psi0*parin.R0*(1-parin.beta0)*hf(1,:));
prof.dF2dpsi_vec= 2*C0*sqrt(psi0)*parin.R0*(1-parin.beta0)*hf_prime(1,:); % not
prof.dpdpsi_vec= prof.p0*hp_prime(1,:);
% prof.q_vec = prof.F_vec./(2*pi).*trapz(theta_vec, ...
%   sqrt(R_th.^2+Z_th.^2)./(R.^2.* prof.Bp));
prof.q_vec = prof.F_vec./(2*pi).*sum(theta_wi.*sqrt( ...
  R_th.^2+Z_th.^2)./(R.^2.* prof.Bp));

% prof.Vp = 2*pi*trapz(theta_vec, trapz(rho_vec, J.*R, 2));
prof.Vp_vec = 2*pi*cumsum(sum(theta_wi.*rho_wi.*J.*R));
% prof.Vp = 2*pi*sum(sum(theta_wi.*rho_wi.*J.*R));
prof.Vp=prof.Vp_vec(end);
% prof.Bp2_vavg = 2*pi*trapz(theta_vec, trapz(rho_vec, ...
%   J.*R.*prof.Bp.^2, 2))/prof.Vp;
prof.Bp2_vavg=2*pi*sum(sum(theta_wi.*rho_wi.*J.*R.*prof.Bp.^2))/prof.Vp;
% prof.kavg_vec=1/(2*pi*parin.a)*trapz(theta_vec, ...
%   sqrt(R_th.^2+Z_th.^2));
prof.kavg_vec=1/(2*pi*parin.a)*sum( ...
  theta_wi.*sqrt(R_th.^2+Z_th.^2));
% prof.pavg=2*pi/prof.Vp*trapz(theta_vec, trapz(rho_vec, prof.p.*J.*R, 2));
prof.pavg=2*pi/prof.Vp*sum(sum(prof.p.*J.*R.*rho_wi.*theta_wi));
% prof.Bpavg=mu0*parin.Ip/(2*pi*parin.a*prof.kavg_vec);
% prof.Lp_vec=trapz(theta_vec, sqrt(R_th.^2+Z_th.^2));
prof.Lp_vec=sum(theta_wi.*sqrt(R_th.^2+Z_th.^2));
prof.Lp=prof.Lp_vec(end);
% prof.Sp_vec=trapz(theta_vec,2*pi*R.*sqrt(R_th.^2+Z_th.^2)); % to check
% prof.Sp_vec=2*pi*sum(theta_wi.*R.*sqrt(R_th.^2+Z_th.^2)); % to check
% prof.Sp_vec=trapz(theta_vec, J);
prof.Sp_vec=cumsum(sum(theta_wi.*rho_wi.*J)); % 25-03-08 13:14
prof.Sp=prof.Sp_vec(end);
% prof.Bpavg_vec=trapz(theta_vec,prof.Bp.*sqrt(R_th.^2+ ...
%   Z_th.^2))./prof.Lp_vec;
prof.Bpavg_vec=sum(theta_wi.*prof.Bp.*sqrt(R_th.^2+ ...
  Z_th.^2))./prof.Lp_vec;
prof.Bpa=prof.Bpavg_vec(end);
prof.Bpa2=mu0*parin.Ip/prof.Lp;
prof.betat=2*mu0*prof.pavg/parin.B0^2;
prof.betaN=100*prof.betat*parin.a*parin.B0/(parin.Ip*1e-6);
prof.betap=prof.betat*(parin.B0/prof.Bpa)^2;
prof.li=prof.Bp2_vavg/prof.Bpa.^2; % Internal inductance
prof.Wk=1.5*prof.pavg*prof.Vp; % kinetic energy, 3/2*p_avg*Vp
prof.qi=parin.a*parin.B0/(parin.R0*prof.Bpa); % kink (Sykes) safety factor
% prof.q95=interp1(prof.psi_vec,prof.q_vec,0.95); % safety factor at psi=0.95

LHStmp1=(Z_th.*grad_psi_R-R_th.*grad_psi_Z)./R;
LHStmp2=(Z_ro.*grad_psi_R-R_ro.*grad_psi_Z)./R;
[LHStmp1_rho, ~] = gradient(LHStmp1, ...
  rho(1,2)-rho(1,1), theta(2,1)-theta(1,1));
[~, LHStmp2_theta] = gradient(LHStmp2, ...
  rho(1,2)-rho(1,1), theta(2,1)-theta(1,1));
prof.LHS=(LHStmp1_rho-LHStmp2_theta)*psi0;
prof.RHS=-mu0*prof.jphi.*J;

prof.Phi_vec=cumsum(prof.F_vec.*sum(rho_wi.*theta_wi.*J./(R.^1))); % 25-03-07 13:00 to check
prof.PhiB=prof.Phi_vec(end);
prof.rhotn_vec=sqrt(prof.Phi_vec/prof.PhiB);

prof.q_vec2=prof.F_vec./(2*pi).*sum(theta_wi.*J./R)./prof.psi_rho_vec;
% 25-03-08 07:15 to check Jacobian J_psi -> J_ro
prof.c0_vec=sum(theta_wi.*J)./prof.psi_rho_vec;
prof.c1_vec=sum(theta_wi.*J.*R)./prof.psi_rho_vec;
prof.c2_vec=sum(theta_wi.*J./R)./prof.psi_rho_vec;
prof.c3_vec=sum(theta_wi.*J.*R.*prof.Bp.^2)./prof.psi_rho_vec;
prof.c4_vec=sum(theta_wi.*J.*R.^3.*prof.Bp.^2)./prof.psi_rho_vec;
prof.dVdpsi_vec=2*pi*prof.c1_vec;
prof.g1_vec=4*pi^2*prof.c1_vec.*prof.c4_vec;
prof.g2_vec=4*pi^2*prof.c1_vec.*prof.c3_vec;
prof.g3_vec=prof.c2_vec./prof.c1_vec;
prof.Ip_vec=1/(4*pi^2*mu0).*prof.F_vec.*prof.g2_vec.*prof.g3_vec./prof.q_vec/(2*pi);
prof.dVdrhotn_vec=4*pi*prof.PhiB*prof.c1_vec./( ...
  prof.F_vec.*prof.c2_vec).*prof.rhotn_vec;

end


