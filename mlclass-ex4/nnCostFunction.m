function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1=[ones(m,1) X]; %add b0 column
ncol = size(X,2) ;
hxhidden=sigmoid(a1*Theta1') ;
a2=[ones(m,1) hxhidden]; %add b0 column
%fprintf('hidden layer : %d %d\n', size(a2));
%%%layer2
h3= sigmoid(a2*Theta2');
%fprintf('layer3 size:%d %d\n', size(h3)); 
% 5000 10: This is the k category of y
%%%%%k categories of y
part=zeros(1,num_labels);
delta3=zeros(m,num_labels);
for c=1:num_labels
yk=(y==c); 
pos=find(y==c);
neg=find(y~=c);
part(c)= -yk(pos)'* log(h3(pos,c))-(1-yk(neg))'*log(1-h3(neg,c)) ;
%%gradient
delta3(:,c)= h3(:,c) -yk;
end;

J= sum(part)/m +lambda* sum([sum(Theta1(:,2:end).^2)(:);sum(Theta2(:,2:end).^2)(:)] )/(2*m);


% dimen of delta2 : (nrow of X) * (ncol of layer2)
%% Be noted of the dot product between g'(a2) and delta2*Theta2
delta2=  (a2.*(1.-a2)).* (delta3*Theta2) ;
%Theta2_grad = (Theta2_grad + delta3'*a2)/m;

%%%regularize item
reg2=[zeros(size(Theta2,1),1) lambda*Theta2(:,2:end)];
Theta2_grad = (Theta2_grad + delta3'*a2+reg2)/m;

delta2=delta2(:,2:end); % remove beta0 column
%%regularize item
reg1=[zeros(size(Theta1,1),1) lambda*Theta1(:,2:end)];

Theta1_grad =  (Theta1_grad + delta2'*a1+reg1)/m;

%theta = theta -  X'*(h3-y)/m;
%grad(1) =  X(:,1)'*(h3-y)/m;

%grad(2:size(grad)) =  X(:,2:ncol)'*(h3-yk)/m +lambda*theta%(2:length(theta))/m;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
