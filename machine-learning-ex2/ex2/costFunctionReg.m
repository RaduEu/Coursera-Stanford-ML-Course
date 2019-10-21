function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

Jtemp = 0;
for i = 1:m
	Jtemp +=-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta));
endfor
for i = 2:size(theta)
	Jtemp+=lambda/2*theta(i)^2;
endfor
J = Jtemp/m;

Gtemp = zeros(size(theta));
for i = 1:size(theta)
	for j = 1:m
		Gtemp(i) += (sigmoid(X(j,:)*theta)-y(j))*X(j,i);
		endfor
endfor
for i = 2:size(theta) Gtemp(i)+=lambda*theta(i);
endfor
for i = 1:size(theta) Gtemp(i)/=m;
endfor
grad = Gtemp;


% =============================================================

end
