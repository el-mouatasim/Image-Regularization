% you need to dowload HNO master in github
clear all
close all
addpath ./../HNO
%addpath ./../image
%%
%X=double(imread('man.tiff'));
%X=double(imread('boat.png'));
%X=double(imread('lena_512.tif'));
X=double(imread('Cameraman_512.tif'));
%X=double(imread('Goldhill_512.gif'));
%X=double(imread('patches.gif'));
%%
[m, n] = size(X);
%% control param
 ni=1.2; %ni>1
 nd=0.7;  %nd <1 
 n1=1.6; % n1 > 1
%%
[P,center]=psfGauss([9,9],4);
B=imfilter(X,P,'symmetric');
randn('seed',314);
y=B +(1e-4)*randn(size(B));
%%

clear pars
pars.mon=0;
pars.MAXITER=250;
pars.denoiseiter=5;
pars.n1=n1;
pars.ni=ni;
pars.nd=nd;
pars.true=X;
stopcriterion=1;
tolerance=1*10^(-4);
lambda = 0.001;
%%
[X_ACPG,objective_ACPG, times_ACPG, mses_ACPG]=deblur_l1_CPGA(y,P,center,lambda,stopcriterion,tolerance,X,pars);
mse_ACPG = norm(X-X_ACPG,'fro')^2 /(m*n);
ISNR_ACPG = 10*log10( norm(y-X,'fro')^2 / (mse_ACPG*m*n) );
cpu_time_ACPG = times_ACPG(end);
%

%%
[X_CSA,objective_CSA, times_CSA, mses_CSA]=deblur_l1_CSA(y,P,center,lambda,stopcriterion,tolerance,X,pars);
mse_CSA = norm(X-X_CSA,'fro')^2 /(m*n);
ISNR_CSA = 10*log10( norm(y-X,'fro')^2 / (mse_CSA*m*n) );
cpu_time_CSA = times_CSA(end);
%%
stopcriterion=3;
tolerance=objective_ACPG(end);
[X_FISTA,objective_FISTA, times_FISTA, mses_FISTA]=deblur_l1_FISTA_m(y,P,center,lambda,stopcriterion,tolerance,X,pars);
mse_FISTA = norm(X-X_FISTA,'fro')^2 /(m*n);
ISNR_FISTA = 10*log10( norm(y-X,'fro')^2 / (mse_FISTA*m*n) );
cpu_time_FISTA = times_FISTA(end);

%%

%%
%%%% display results and plots
fprintf('FISTA CPU time = %3.3g seconds,  \titers = %d \tMSE = %3.3g, ISNR = %3.3g dB\n', cpu_time_FISTA, length(objective_FISTA), mse_FISTA, ISNR_FISTA)
fprintf('CSA\n CPU time = %3.3g seconds,  \titers = %d \tMSE =%3.3g, ISNR = %3.3g dB\n', cpu_time_CSA, length(objective_CSA), mse_CSA, ISNR_CSA)
fprintf('ACPG\n CPU time = %3.3g seconds,  \titers = %d \tMSE =%3.3g, ISNR = %3.3g dB\n', cpu_time_ACPG, length(objective_ACPG), mse_ACPG, ISNR_ACPG)

%%%%%%%%%
 figure; 
 subplot(3,2,1)
 imagesc(X), colormap gray, axis off, axis equal
 title('Original')
% 
 subplot(3,2,2)
 imagesc(y), colormap gray, axis off, axis equal
 title('Blurred and noisy')
% 
subplot(3,2,3)
imagesc(X_FISTA), colormap gray, axis off; axis equal,
title('Estimated using FISTA')
% 
subplot(3,2,4)
imagesc(X_CSA), colormap gray, axis off; axis equal,
title('Estimated using CSA')
% % 
subplot(3,2,5)
imagesc(X_ACPG), colormap gray, axis off; axis equal,
title('Estimated using CPGA')

figure;
subplot(2,2,1)
  semilogy(times_FISTA,objective_FISTA, 'b', 'LineWidth',1.8), hold on, 
  semilogy(times_CSA,objective_CSA, 'g', 'LineWidth',1.8), hold on,
  semilogy(times_ACPG, objective_ACPG,'r--', 'LineWidth',1.8),
  title('Objective function 0.5||y-Ax||_{2}^{2}+\tau||x||_1','FontName','Times','FontSize',12),
  set(gca,'FontName','Times'),
  set(gca,'FontSize',12),
  xlabel('seconds'), 
  legend('FISTA','CSA','CPGA');
subplot(2,2,2)
  semilogy(1:length(objective_FISTA),objective_FISTA, 'b', 'LineWidth',1.8), hold on, 
  semilogy(1:length(objective_CSA),objective_CSA, 'g', 'LineWidth',1.8), hold on,
  semilogy(1:length(objective_ACPG), objective_ACPG,'r--', 'LineWidth',1.8),
  title('Objective function 0.5||y-Ax||_{2}^{2}+\tau||x||_1','FontName','Times','FontSize',12),
  set(gca,'FontName','Times'),
  set(gca,'FontSize',12),
  xlabel('iteration'), 
  legend('FISTA','CSA','ACPG');
% % 
