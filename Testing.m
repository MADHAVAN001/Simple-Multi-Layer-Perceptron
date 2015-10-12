%full training using the model after obtaining the number of neurons from
%the validation test

%New test data created to plot the function

a = 1.0;
b = 0.5;
learning_factor = 0.06;
k = -1;
train = [-1;-0.9;-0.8;-0.7;-0.6;-0.5;-0.4;-0.3;-0.2;-0.1;
0;0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9;1;];

train_output = 0.8 * sin(pi*train);
neuron_selection_error = zeros(8,1);
mse_error = zeros(5000,1);

min_error = 3;
test = zeros(16,1);
for i = 1:16
    test(i) = -1 + 2/15 * (i-1);
end
%test = train;
test_desired_output = 0.8 * sin(pi*test);
test_actual_output = zeros(16,1);

num_neurons = 8;
w = randi(100,2,num_neurons-1)/100;
y = zeros(1,num_neurons);
w_second = randi(100,1,num_neurons)/50;
for i = 1:5000
    for j = 1:21
        z(1,1) = train(j);
        z(2,1) = -1;
        t(1,:) = w(1,:) * z(1);
        t(2,:) = w(2,:) * z(2);
        d = train_output(j);
        
        %forward phase - to 1st hidden layer
        w_sum = sum(t,1);
        for k = 1:num_neurons-1
            y(k) = a*(-1+2/(1+exp(-b*w_sum(k))));
        end
        
        y(num_neurons) = -1;
        
        %forward phase - from hidden layer to output layer
        u = sum(y*w_second',2);
        output = a*(-1+2/(1+exp(-b*u)));
        desired_output = 0.8*sin(3.14*train(j));
        
        %calculation of f_dash
        f_dash = a*b*(1-output*output/a*a)/2;
        f_dash_first = zeros(num_neurons,1);
        for k = 1:num_neurons
            f_dash_first(k) = a*b*(1-y(k)*y(k)/a*a)/2;
        end
        f_dash_first(num_neurons) = 0;
        
        %back propagation from output layer to hidden layer
        error = 0.5*(d-output)*(d-output);
        delta_op = (d-output) * f_dash;
        change_w_second = (learning_factor * delta_op) * y;
        
        
        w_second = w_second + change_w_second;
        %back propagation from hidden layer to input layer
        delta_y = (f_dash_first .* (w_second'*delta_op));
        delta_y = delta_y(1:num_neurons-1,:);
        change_prev_layer = learning_factor * (delta_y * z');
        
        %updating the weights of both the layers
        w = w + change_prev_layer';
    end
    
    %calculating validation error
    for j = 1:16
        t(1,:) = w(1,:) * test(j);
        t(2,:) = w(2,:) * -1;
        d = test_desired_output(j);
        
        %forward phase - to 1st hidden layer
        w_sum = sum(t,1);
        for k = 1:num_neurons-1
            y(k) = a*(-1+2/(1+exp(-b*w_sum(k))));
        end
        y(num_neurons) = -1;
        
        %forward phase - from hidden layer to output layer
        u = sum(y.*w_second,2);
        test_actual_output(j) = a*(-1+2/(1+exp(-b*u)));
        
        mse_error(i) = mse_error(i) + ((d-test_actual_output(j))^2);
    end
    
    %stopping criterion
    if i>1 && min_error > mse_error(i)
        op = test_actual_output;
        min_error = mse_error(i);
    end
end
figure;
plot(test,test_actual_output, test, test_desired_output);