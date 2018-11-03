function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

cost_main = -1.*y.*log(sigmoid(X*theta)) -(1-y).*log(1-sigmoid(X*theta));
cost_main = sum(cost_main)/m;
reg = 0.5.*sum(theta(2:n).^2)*lambda./m;
J = cost_main + reg;

gradIntm = sigmoid(X*theta) - y;
grad0 = sum(gradIntm)/m;

gradOther = ((X(:,2:n))' * gradIntm)/m + lambda .* theta(2:n)./m;

grad = [grad0;gradOther];
% =============================================================

end
