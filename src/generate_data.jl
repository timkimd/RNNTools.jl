function generate_data_copy_task(args)
    # for copy task
    data = randn(args.obs_dim+1, args.ntrials, args.ntimebins) .> 0
    data[:,:,(args.ntimebins÷2)+1:end] .= 0
    data[args.obs_dim+1,:,1:(args.ntimebins÷2)] .= 0
    data[args.obs_dim+1,:,(args.ntimebins÷2)+1:end] .= 1
    target = randn(args.obs_dim, args.ntrials, args.ntimebins) .> 0
    target[:,:,1:(args.ntimebins÷2)] .= 0
    target[:,:,(args.ntimebins÷2)+1:end] .= data[1:args.obs_dim,:,1:(args.ntimebins÷2)]
    return Float32.(data), Float32.(target)
end

function generate_data_n_bit_task(args)
    # n-bit task
    data = zeros(args.obs_dim, args.ntrials, args.ntimebins)
    target = zeros(args.obs_dim, args.ntrials, args.ntimebins)
    prev = ones(args.obs_dim, args.ntrials)
    for i = 1:args.ntrials
        npulses = rand(Poisson(args.frequency))
        tstart = rand([1:args.ntimebins;], npulses)
        for j = 1:npulses
            if tstart[j] <= ( args.ntimebins - args.width )
                ch = rand([1:args.obs_dim;]) # pick one channel
                mg = sign(randn()) # plus one or minus one
                data[ch, i, tstart[j]:tstart[j]+args.width] .= mg
            end 
        end
    end

    for t = 1:args.ntimebins
        changed = data[:,:,t] .!= 0
        prev[changed] = data[changed,t]
        target[:, :, t] = prev
    end

    return Float32.(data), Float32.(target)
end

function generate_data_n_bit_task_varying_amp(args)
    # n-bit task
    data = zeros(args.obs_dim, args.ntrials, args.ntimebins)
    target = zeros(args.obs_dim, args.ntrials, args.ntimebins)
    prev = zeros(args.obs_dim, args.ntrials)
    for i = 1:args.ntrials
        npulses = rand(Poisson(args.frequency))
        tstart = rand([1:args.ntimebins;], npulses)
        for j = 1:npulses
            if tstart[j] <= ( args.ntimebins - args.width )
                ch = rand([1:args.obs_dim;]) # pick one channel
                mg = 2 * rand() - 1 # U[-1, 1]
                data[ch, i, tstart[j]:tstart[j]+args.width] .= mg
            end 
        end
    end

    for t = 1:args.ntimebins
        changed = data[:,:,t] .!= 0
        prev[changed] = data[changed,t]
        target[:, :, t] = prev
    end

    return Float32.(data), Float32.(target)
end

function generate_data_n_bit_task_rectangle(args)
    # n-bit task; n = 2
    data = zeros(args.obs_dim, args.ntrials, args.ntimebins)
    target = zeros(args.obs_dim, args.ntrials, args.ntimebins)
    prev = zeros(args.obs_dim, args.ntrials)
    for i = 1:args.ntrials
        npulses = rand(Poisson(args.frequency))
        tstart = rand([1:args.ntimebins;], npulses)
        for j = 1:npulses
            if tstart[j] <= ( args.ntimebins - args.width )
                ch = rand([1:args.obs_dim;]) # pick one channel
                if ch == 1
                    mg = 4 * rand() - 2 # U[-1, 1]
                else
                    mg = 2 * rand() - 1 # U[-1, 1]
                end
                data[ch, i, tstart[j]:tstart[j]+args.width] .= mg
            end 
        end
    end

    for t = 1:args.ntimebins
        changed = data[:,:,t] .!= 0
        prev[changed] = data[changed,t]
        target[:, :, t] = prev
    end

    return Float32.(data), Float32.(target)
end

function generate_data_n_bit_task_disk(args)
    # n-bit task; n = 2
    data = zeros(args.obs_dim, args.ntrials, args.ntimebins)
    target = zeros(args.obs_dim, args.ntrials, args.ntimebins)
    prev = zeros(args.obs_dim, args.ntrials) # something
    prev[1,:] .= 1.5 # start somewhere other than 0
    prev[2,:] .= 0
    for i = 1:args.ntrials
        npulses = rand(Poisson(args.frequency))
        tstart = rand([1:args.ntimebins;], npulses)
        for j = 1:npulses
            if tstart[j] <= ( args.ntimebins - args.width )
                mg = 4 * rand(2) .- 2
                while !(1 < norm(mg) < 2)
                    mg = 4 * rand(2) .- 2
                end
                data[:, i, tstart[j]:tstart[j]+args.width] .= mg
            end 
        end
    end

    for t = 1:args.ntimebins
        changed = data[:,:,t] .!= 0
        prev[changed] = data[changed,t]
        target[:, :, t] = prev
    end

    return Float32.(data), Float32.(target)
end

function generate_data_n_bit_task_ring(args)
    # n-bit task; n = 2
    data = zeros(args.obs_dim, args.ntrials, args.ntimebins)
    target = zeros(args.obs_dim, args.ntrials, args.ntimebins)
    prev = zeros(args.obs_dim, args.ntrials)
    prev[1,:] .= 2 # start somewhere other than 0
    prev[2,:] .= 0
    for i = 1:args.ntrials
        npulses = rand(Poisson(args.frequency))
        tstart = rand([1:args.ntimebins;], npulses)
        for j = 1:npulses
            if tstart[j] <= ( args.ntimebins - args.width )
                angle = 2 * π * rand()
                x = 2 * cos(angle)
                y = 2 * sin(angle)
                mg = [x;y]
                data[:, i, tstart[j]:tstart[j]+args.width] .= mg
            end 
        end
    end

    for t = 1:args.ntimebins
        changed = data[:,:,t] .!= 0
        prev[changed] = data[changed,t]
        target[:, :, t] = prev
    end

    return Float32.(data), Float32.(target)
end