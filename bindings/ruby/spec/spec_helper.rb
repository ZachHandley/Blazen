# frozen_string_literal: true

require "blazen"

RSpec.configure do |config|
  config.expect_with :rspec do |c|
    c.syntax = :expect
  end
  config.example_status_persistence_file_path = ".rspec_status"
  config.disable_monkey_patching!
  config.shared_context_metadata_behavior = :apply_to_host_groups
end
