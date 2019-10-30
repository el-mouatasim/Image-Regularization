%%% This is a modified version of the file deblur_CSA_l1
%+subgrad function with step size control;


function [x1, objective, times, mses] = CSA0(A,y,tau,stopcriterion,tolerance, maxiters,n1,ni,nd,true,verbose)

%y1 = zeros(size(AT(y)));
%z = AT(y);
%z = randn(size(y1));
times(1) = 0;
t0 = cputime;

%%%%%%%%%%%%%%%%%%=============Nestervo's algorithm=================%%%%%%%%%%%%%%%%%%

 a1=1;
 
% 
x0=y;
%Wx = W(x0);
resid =  y-A(x0);
objective(1)=0.5*(resid(:)'*resid(:)) + tau*sum(sum( abs(x0) ));
g1=subgrad(x0,y,A,AT,tau);
%subgradient
err = x0-true;
mses(1) = (err(:)'*err(:));
if (verbose)
    fprintf('iter = %d, obj = %3.3g\n', 1, objective(1))
end
% Compute and store initial value of the objective function

% 
n_dec=nd; %0<n_dec<1  
n_inc=ni; % 1<n_inc mse=148(1.8)
%sped subgrdient
% n_dec=0.35;  
% n_inc=1.85; 
% n_1=1.15; %

for k = 2:maxiters

   alpha0=n1;
   x1=y1-alpha0*g1;
   %  >>>>>>>>>  nouveau point
    %Wx = W(x1);
    resid =  y-A(x1);
    objective(k)=0.5*(resid(:)'*resid(:)) + tau*sum(sum( abs(x1) ));
    err = x1-true;
    mses(k) =  (err(:)'*err(:));
    %
    a2=(1+sqrt(4*a1^2+1))/2;
    y2=x1+(a1-1)*(x1-x0)/a2;
    g1=subgrad(y2,y,A,AT,tau);
    %subgradient
    if objective(k) > objective(k-1) 
       n2 = n_dec*n1;
    elseif k==2
        n2 = 0.5*n1;
    else
       n2=n_inc*n1;
   end
    x0=x1;
    a1=a2;
    y1=y2;
    n1=n2;
    
    times(k) = cputime - t0;
           
    switch (stopcriterion)
        case 1
            criterion = abs(objective(k)-objective(k-1))/objective(k);  
        case 2
            criterion = norm(x1-x0,'fro')/sqrt(sum(sum( x1.^2 )));
        case 3
            criterion = objective(k);
      %  case 4
      %      criterion = norm(g1,'fro');
        otherwise
            error('Invalid stopping criterion!');
    end
    
    if (verbose)
        fprintf('iter = %d, obj = %3.3g, stop criterion = %3.3g, ( target = %3.3g )\n', k, objective(k), criterion, tolerance)
    end
    
    if (criterion < tolerance)
        break;
    end
   
end

