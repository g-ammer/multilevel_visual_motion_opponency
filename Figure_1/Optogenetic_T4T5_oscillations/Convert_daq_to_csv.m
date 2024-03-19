
listing = dir('*.daq');

length(listing)

for k = 1:length(listing)
  name = listing(k).name;
  
  data = daqread(name);
  data2 = data(:,1:2);
  data2(:,1) = data2(:,1)/20;
  
  writematrix(data2, strrep(name,'.daq','.csv'));
  
end

plot(data2)

