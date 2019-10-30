function [X_out,fun_val, times, mses]=deblur_l1_CSA(Bobs,P,center,lambda,stopcriterion,tolerance,true,pars)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function implements FISTA for solving the linear inverse problem with 
% an orthogonal l1 wavelet regularizer and either reflexive or periodic boundary
% conditions
%
% Based on the paper
% Amir Beck and Marc Teboulle, "A Fast Iterative Shrinkage-Threshold Algorithm
% for Linear Inverse Problems",  to appear in SIAM Journal on Imaging
% Sciences
% -----------------------------------------------------------------------
% Copyright (2008): Amir Beck and Marc Teboulle
% 
% FISTA is distributed under the terms of 
% the GNU General Public License 2.0.
% 
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
% 
% INPUT
%
% Bobs............................. The observed image which is blurred and noisy
% P .................................... PSF of the blurring operator
% center ......................  A vector of length 2 containing the center
%                                           of the PSF
% W ....................................  A function handle. For an image
%                                           X, W(X)  is  an orthogonal
%                                           transform of the image X.
% WT .................................  A function handle. For an image
%                                           X, WT(X) is the inverse (with respect to the operator W) 
%                                           orthogonal transform of the image X 
% lambda ...................... Regularization parameter
% pars.................................Parameters structure
% pars.MAXITER ..................... maximum number of iterations
%                                                      (Default=100)
% pars.fig ............................... 1 if the image is shown at each
%                                                      iteration, 0 otherwise (Default=1)
% pars.BC .................................. boundary conditions.
%                                                      'reflexive'  (default)  or 'periodic
% OUTPUT
% 
% X_out ......................... Solution of the problem
%                                          min{||A(X)-Bobs||^2+lambda \|Wx\|_1
% fun_all .................... Array containing all function values
%                                          obtained in the FISTA method


% Assigning parameters according to pars and/or default values
flag=exist('pars');
if (flag&isfield(pars,'MAXITER'))
    MAXITER=pars.MAXITER;
else
    MAXITER=100;
end
if(flag&isfield(pars,'fig'))
    fig=pars.fig;
else
    fig=1;
end
if (flag&isfield(pars,'BC'))
    BC=pars.BC;
else
    BC='reflexive';
end
if (flag&isfield(pars,'n1'))
    n1 = pars.n1;
end
if (flag&isfield(pars,'ni'))
    ni = pars.ni;
end
if (flag&isfield(pars,'nd'))
    nd = pars.nd;
end
if (flag&isfield(pars,'true'))
    true = pars.true;
end


% If there are two output arguments, initalize the function values vector.
if (nargout==2)
    fun_all=[];
end
%
times(1) = 0;
t0 = cputime;
%
[m,n]=size(Bobs);
Pbig=padPSF(P,[m,n]);

switch BC
    case 'reflexive'
        trans=@(X)dct2(X);
        itrans=@(X)idct2(X);
        % computng the eigenvalues of the blurring matrix         
        e1=zeros(m,n);
        e1(1,1)=1;
        Sbig=dct2(dctshift(Pbig,center))./dct2(e1);
    case 'periodic'
        trans=@(X) 1/sqrt(m*n)*fft2(X);
        itrans=@(X) sqrt(m*n)*ifft2(X);
        % computng the eigenvalues of the blurring matrix         
        Sbig=fft2(circshift(Pbig,1-center));
    otherwise
        error('Invalid boundary conditions should be reflexive or periodic');
end
% computing the two dimensional transform of Bobs
Btrans=trans(Bobs);

%The Lipschitz constant of the gradient of ||A(X)-Bobs||^2
%L=2*max(max(abs(Sbig).^2));

% initialization
X_iter=Bobs;
Y=X_iter;
t_new=1;
fprintf('************\n');
fprintf('**CSA**\n');
fprintf('************\n');
%fprintf('#iter  fun-val         relative-dif\n==============================\n');
for i=1:MAXITER
    alpha = n1;
    % Store the old value of the iterate and the t-constant
   X_old=X_iter;
   t_old=t_new;
   
    % Gradient step
    D=Sbig.*trans(Y)-Btrans;
    g1=2*itrans(conj(Sbig).*D);
    g1=g1+lambda*sign(Y);
    g1=real(g1);
     
    % The new iterate 
    X_iter=Y-alpha*g1;
    
    
    %updating t and Y
    t_new=(1+sqrt(1+4*t_old^2))/2;
    Y=X_iter+(t_old-1)/t_new*(X_iter-X_old);
    
    % Compute the l1 norm of the wavelet transform and the function value and store it in
    % the function values vector fun_all if exists.
    t=sum(sum(abs(X_iter)));
    fun_val(i)=norm(Sbig.*trans(X_iter)-Btrans,'fro')^2+lambda*t;
   if  i==2 &&  fun_val(i) <= fun_val(i-1)
            n2 = 0.5*n1;
    elseif i>1 && fun_val(i) > fun_val(i-1)
            n2 = nd*n1;
    else
            n2=ni*n1;
    end
     n1=n2;
    err = X_iter-true;
    mses(i) =  (err(:)'*err(:));
    times(i) = cputime - t0;
        if i>1
        switch (stopcriterion)
        case 1
            criterion = abs(fun_val(i)-fun_val(i-1))/fun_val(i);  
        case 2
            criterion = norm(X_iter-X_old,'fro')/sqrt(sum(sum( X_iter.^2 )));
        case 3
            criterion = fun_val(i);
        otherwise
            error('Invalid stopping criterion!');
        end
        if (criterion < tolerance )%lena -0.1 -0.4
        break;
        end     
        end

end

X_out=X_iter;
