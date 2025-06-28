arr = repmat(1:4, 1, 20);  % Repeat 1:4 five times
s = arr(randperm(length(arr)));
a = s;
Isa = mutual_information_basic(s(2:end),a(2:end),0.1)
Iss1 = mutual_information_basic(s(2:end),s(1:end-1),0.1)
%Isa_s1 = cond_mutual_information(s(tdx:end),a(tdx:end),s(tdx-1:end-1),0.1)
Iss1_a = cond_mutual_information(s(tdx:end),s(tdx-1:end-1),s(tdx:end),0.1)

Isa_s1_check = Isa - Iss1 + Iss1_a

s = [4, 3, 3, 4, 2, 1, 4, 3, 3, 2, 1, 4, 4, 2, 1, 3, 3, 2, 1, 3,...
2, 1, 4, 3, 2, 1, 4, 3, 3, 4, 2, 1, 3, 4, 2, 1, 3, 4, 2, 1,...
2, 1, 4, 3, 4, 2, 1, 3, 2, 1, 3, 4, 3, 4, 2, 1, 3, 4, 3, 4,...
2, 1, 2, 1, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 3, 4, 2, 1, 3, 4];
a = s;
Isa = mutual_information_basic(s(2:end),a(2:end),0.1)
Iss1 = mutual_information_basic(s(2:end),s(1:end-1),0.1)
%Isa_s1 = cond_mutual_information(s(tdx:end),a(tdx:end),s(tdx-1:end-1),0.1)
Iss1_a = cond_mutual_information(s(tdx:end),s(tdx-1:end-1),s(tdx:end),0.1)

Isa_s1_check = Isa - Iss1 + Iss1_a


s=[4, 4, 2, 1, 3, 2, 1, 4, 3, 3, 2, 1, 3, 4, 2, 1, 3, 4, 4, 2,...
1, 3, 4, 3, 4, 2, 1, 2, 1, 2, 1, 2, 1, 3, 4, 4, 3, 2, 1, 2,...
1, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 2, 1, 4, 3,...
2, 1, 3, 4, 2, 1, 3, 4, 4, 3, 3, 4, 2, 1, 2, 1, 2, 1, 4, 3]
a = s;
Isa = mutual_information_basic(s(2:end),a(2:end),0.1)
Iss1 = mutual_information_basic(s(2:end),s(1:end-1),0.1)
%Isa_s1 = cond_mutual_information(s(tdx:end),a(tdx:end),s(tdx-1:end-1),0.1)
Iss1_a = cond_mutual_information(s(tdx:end),s(tdx-1:end-1),a(tdx:end),0.1)

Isa_s1_check = Isa - Iss1 + Iss1_a