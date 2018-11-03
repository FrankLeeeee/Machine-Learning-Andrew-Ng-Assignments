function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
%   Theta1 is 25*401 matrix here
%   Theta2 is 10*26 matrix here
%   X is 5000*401 matrix here
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
x1 = [ones(m,1) X];
z2 = x1*Theta1'; % give a 5000*25 matrix
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];; % give a 5000*26 matrix
z3 = a2*Theta2'; % give a 5000*10 matrix
a3 = sigmoid(z3);
[Y,p] = max(a3,[],2);

% =========================================================================


end
